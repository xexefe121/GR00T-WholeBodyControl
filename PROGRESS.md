# G1 true23 SONIC — progress log

## 2026-09-10: new compact native23 learner implemented; continuous main training started

Live-state correction21:57 Sydney:main73693 FAILED after5completed updates
during next rollout; no longer running. Cause was fixed3e-5 CPU/GPU base-reward
summation audit threshold; actual returned/stored rewards matched exactly.
Version4 replaces that diagnostic with a derived float32 reduction error bound,
not a physical/reward relaxation. V4 smoke89841 EXIT0,384transitions/32Adamsteps;
independent CPU16state audit passes(goal4.77e-7,packedexact).
Fresh main32479 now launched in train2000_v2 with same planned2000updates.
Driver train_g1_true23_compact_tracker_v4.py. Check THIS job before restart;
train2000_v1 is retained failed evidence and cannot resume its lost update5.

User renewed twelve-hour target at21:35 Sydney: full-body PICO teleop, including
both legs. Goal remains ACTIVE. Deployment NOT READY. No robot commands.

Reviewed prior controller/training failures, original23dofsonic task, transfer
method, and exact saved training-state audit. Failing PICO phase is represented:
foot1000 visited all5780 source anchors; queried +/-25-frame neighborhoods have
3883–6754 transitions. Missing-phase and repeated reward/reset tweaks are not
supported. Latest continuous frozen-decoder lineage has2.048m transitions.

Distinct new route: preserve SONIC/PICO source motion, train compact native23
reference-residual tracker directly.180014 actor parameters,165 inputs,23
physical outputs. This is motion-compatible, NOT frozen-SONIC-policy parity.
Same native physics, source timing, original29 hand/head targets, both feet,
bounded reward/foot bonus, joint/effort/velocity bounds and full replay referee.

7 focused tests pass. Two setup-only config failures retained(smoke_v1/v2,
zero optimizer updates). Version3 smoke72666 EXIT0:384transitions,32Adam steps;
all8MLP tensors updated. Every smoke reward/timeout bootstrap checked:returned
and stored error0,base summation7.63e-6. Independent CPU input audit EXIT0:
16actual states,goal90 max4.77e-7,packed165 exact,qroundoff5.96e-8,dq exact.

Main73693 launched:256envs x24controls x2000updates maximum=12.288m transitions,
continuous Adam/critic state; checkpoints0/200/500/1000/2000,4h wall cap.
First complete replay gate at200. Check process/files before any restart.
Run:artifacts/g1_true23_compact_tracker_20260910_v1/train2000_v1.
Driver:gear_sonic/scripts/train_g1_true23_compact_tracker_v3.py.
No automatic extension, acceptance relaxation or hardware promotion.

Zero-update complete-request matrix14595 EXIT0:all4 stop at56controls during
standing(0sourceframes). This records its true untrained starting point; no
standing or motion success claimed. Subsequent checkpoints must learn actual
balance AND improve full-body tracking. See new experiment/failed setup/audit
files; originals and earlier failures preserved, no commit/push.

## 2026-09-10: local ankle fix verified; full-body balance still FAILS—no deployment

Full bounded-torque trial29117 EXIT0; no running job. PICO/002/003/008 complete
2004/1417/1104/681 controls against6530/1417/1569/1114. All4 tracking FAIL.
Exactly one-2.5Nm inward ankle correction at PICO1880 clears old30.60s range
failure. After123 further position-only controls, robot falls at33.08s source:
pelvis.109046m versus target.734578m, tilt1.533699 versus.234855rad; root error
2.072541m. This is the existing absolute-height stop, not a guard/hardware mode
transition. No inference that the single correction alone caused the later fall.

Independent27238 EXIT0:52,060 actual2ms steps/5,206 landmarks/384 policy outputs
exact. Every total PD+correction clip and accepted range prediction checked;
386 fresh10-substep predictions exact, including both torque interventions.
Original4,991 pre-intervention controls/inputs/torques exactly preserved. Actual
range excess0, velocity ratio<=.821186, total effort<=1. Filter max16.999ms and
zero measured>20ms calls; not whole-loop/actual-clock/hardware qualification.
38 tests77037 pass12.87s; scoped Ruff E/F pass. All experiment jobs terminal.

Both new100ms target-only and20ms bounded-torque controllers are REJECTED.
They isolate/repair a local ankle stop but not whole-body balance/foot tracking.
Detailed reproducible result:artifacts/g1_true23_pico_ankle_feedforward_20260910_v1/
OUTCOME.md. Final PICO report1e3d84212233ac23e54507efff92b9231e6d278100302c8e6f42f78d384b6792.
No more ankle parameter sweeps or automatic training extension. Next bounded
work audits actual training-state/reward coverage at this PICO failure phase,
including native leg posture versus original29 torso/hand/foot objectives;
first inspect prior reset/full-decoder/waist experiments, do not assume novelty.
Goal ACTIVE; full-body VR deployment NOT READY. Robot/limits/originals untouched.
Final scoped verification17980 EXIT0:38 tests pass10.83s; all new source/audit
files pass Ruff E/F. Both new controller matrices and every audit are terminal;
no hidden training, pending robot operation or accepted checkpoint remains.

## 2026-09-10:100ms target controller rejected; bounded ankle-torque branch testing

Fixed trial83730 EXIT0: PICO/002/003/008 complete1945/1417/1569/667 controls.
All4 tracking screens FAIL. PICO31.90s gains1.30s over final1000,003 completes
its lifecycle, but relative feet still14–26cm and walking root drift0.94–2.27m.
Independent96360 EXIT0 verifies55,980 actual2ms steps/5,598 landmarks/384 policy
inferences exactly;384 fresh50-substep predictions exact. Filter alone exceeds
20ms on252 controls, max76.82ms. Reject this branch; no horizon sweep/promotion.
Detailed outcome:artifacts/g1_true23_pico_ankle_horizon_20260910_v1/OUTCOME.md.

Dense-force audit47269 EXIT0 independently reconstructs both37,600-step prefixes
and2,000 force partitions without original observer, mj_mulJacTVec or contact-
force helper. Maximum force2.27e-13, acceleration1.36e-12, normal-load residual0.
Fixed torque matrix18097 EXIT0 uses the two previously rejected copied states:
final1000 recovers all23 joint reserves for20ms/100ms with2.5Nm inward ankle
correction; total torque remains under original effort limits. Parent500's
later state fails all0/2.5/5/10Nm probes. Not full-motion or hardware success.

New separately versioned SIM-only ankle_feedforward code retains original20ms
position-only preview first. On its rejection only, test2.5/5/10Nm inward ankle
corrections with unchanged nominal targets/gains and all23 range/velocity checks.
Explicit benchmark clips TOTAL PD+correction to original motor limits every2ms
and records separate torque, without inventing a position-action measurement.
Any exhausted/non-ankle failure latches; no unchecked output or physical path.
38 tests77037 EXIT0/12.87s and Ruff E/F pass. Full fixed PICO/002/003/008 request
is the next evaluation; no outcome claimed yet. No training/source/limit change.
Goal ACTIVE; deployment NOT READY. Robot untouched.

## 2026-09-10: ankle-force cause isolated; anticipatory SIM controller under test

Saved-force capture52871 EXIT0:37,600 original2ms states reproduced exactly;
2,000 actual and80 rejected-probe force samples. Final1000 stop is not motor
effort saturation: requested ankle torque14.38–15.91Nm is below35Nm rating,
but unchanged bounded PD target already supplies maximum available inward
command. Contact loading opposes braking; left foot unloads to0N in the
rejected counterfactual. All8 old search targets are within9.8e-6rad of one
another. Its endpoint inset slightly weakens an already saturated command.
This does not imply that deleting the guard, raising gains or removing target
bounds is safe. Force components hold the solved contact solution fixed.
Evidence:artifacts/g1_true23_pico_ankle_dynamics_20260910_v1/OUTCOME.md.

New isolated g1_true23_ankle_horizon_preview.py retains all-joint20ms checks,
adds same-reserve100ms constant-target checks on all4 ankle axes, preserves
untouched raw bytes and rejects duplicate/non-inward search directions. No
future policy/source input, actual-state reset, model/gain/limit change or
coordinated target optimizer. This is an unqualified anticipatory hypothesis,
not a recursively safe or real-time controller. Existing sources remain pinned.
Focused89744 EXIT0:27 tests pass10.66s; scoped Ruff E/F pass. Fixed saved-state
probes and complete PICO/002/003/008 final1000 evaluations are next; no outcome
or deployment readiness claimed at this entry. No automatic horizon sweep or
new reward-only training. Goal ACTIVE; robot untouched.
Experiment:artifacts/g1_true23_pico_ankle_horizon_20260910_v1/EXPERIMENT.md.

## 2026-09-10: foot continuation COMPLETE; both milestones rejected for deployment

Existing main94600 EXIT0, not stalled:500→1000 preserved-learner updates,
1,024,000 new transitions/4,000 optimizer steps,1351.552214s. Final checkpoint
ac2f844894e3e53c364ae466966a64db556ee255c6ab72e736fe15b87adb19fe.
Audit14325 EXIT0:all rewards/Adam continuation/frozen weights verified. No
automatic extension or qualified export; all experiment jobs now terminal.

Full1000 evaluator54289 EXIT0:002/003/008/PICO complete1417/1104/590/1880
controls against1417/1569/1114/6530. All tracking FAIL. Final PICO30.60/115.60s
is earlier failure than parent50069.34s or milestone60075.14s. On identical
30.60s parent500→1000 improves root p950.354987→0.190386m, leg RMSE
0.162737→0.160060rad, relative feet0.135367/0.163627→0.127001/0.145065m;
still outside required fidelity. Walking003 and008 now stop early too.
PICO1000 stays under30cm root error throughout its executed source, but right
ankle roll approaches upper limit q0.248943,dq+1.347234,margin0.012857rad.
Root drift alone is not sufficient explanation for this checkpoint's stop.

Independent1000 replay/comparison/diagnosis64347 EXIT0:49,910 fresh2ms steps,
4,991 landmark vectors,384 re-inferences exact; original range/effort pass only
on actual prefix.600 has75,540 steps/7,554 landmarks/384 exact re-inferences.
No guard, source, joint, native physics, model or hardware changed.

Actual19,968 sampled training inputs:lower240/VR21 bit-exact; absent slots zero;
3,833 history continuations exact. Initial noiseless orientation audit FAILED,
retained unchanged. Configuration-aware45202 EXIT0 accounts for existing±0.05
orientation/gravity noise, not new training or runtime tolerance. No packing
fault found in sampled fields; measured root velocity remains unaudited.
First/last100-update mean foot RMS0.138272→0.142533m; nonstationary populations,
not evidence of full-motion improvement. No blind reward-only extension.

Video58816 EXIT0:all1881 final PICO states; independently decoded960x720,
50fps,37.62s including7s standing/entry.20s/37.5s frames inspected. Failed SIM,
not hardware footage. See artifacts/g1_true23_pico_foot_precision_20260910_v1/
OUTCOME.md for evidence, comparison and next decision boundary. Goal ACTIVE;
full-body live teleop NOT READY. No robot operation, mode change, commit/push.
Final verification61608 EXIT0:9 focused tests/23.60s and scoped Ruff E/F pass.
Next bounded investigation is saved1000 pre-stop ankle contact/torque behavior;
not another blind reward-only run or repeated old range-termination experiment.

## 2026-09-10: checkpoint600 measured/audited; final1000 training still ACTIVE

Continued existing job94600; no restart or duplicate training. Full600 evaluator
43397 EXIT0. Requests002/003/008/PICO complete1417/1569/461/4107 controls,
against1417/1569/1114/6530 requested. All four tracking screens FAIL.
PICO reaches75.14/115.60 source seconds versus69.34 at parent500, then the
unchanged bounded joint-range preview rejects. Full executed source leg RMSE
0.190101rad; relative-foot p95 0.140899/0.172461m; root p95 1.777880m.
Do not call the longer prefix better full-body tracking:008 now fails after
111/364 source controls and both in-training walks still drift substantially.

Independent600 replay67154 EXIT0:75,540 physical2ms substeps,7,554 measured
landmark vectors,384 policy re-inferences reproduce exactly. Actual prefix
joint range excess0, maximum velocity ratio0.809272, effort ratio<=1.
Evidence consistency does not qualify these failed motions. Evaluation and
replay audit are under artifacts/g1_true23_pico_foot_precision_20260910_v1/eval600_v1.
Fixed main continues toward1000, without changed limits/objective mid-run.
Final1000 evaluation remains required. No physical robot commands.

## 2026-09-10: foot-placement objective implemented; preserved-learner continuation ACTIVE

Previous goal turn classified PROGRESS: completed/audited PICO500 and improved
source duration, without meeting full-body acceptance. This turn re-read current
outcome/trace evidence; old jobs remain terminal. No hardware commands.

Fixed-state encoder probe uses2,048 pairs of actual saved lower-body horizons.
Largest invisible current-leg RMS change0.006588rad (PICO0.001921), not evidence
for losing the observed0.17rad-scale leg motion. Representation sufficiency is
not proved. No encoder replacement, joint deletion or reference bypass added.
Probe:artifacts/g1_true23_pico_leg_feedback_20260910_v1/encoder_resolution_probe.json.

New versioned reward adds independent foot placement precision to existing
rewards:2/(1+normalized_foot_cost/9), zero on any done/timeout.15cm is its reward
scale, NOT a relaxed acceptance threshold. All old rewards, limits, dynamics,
source timing, original29 hand/head intent and actor architecture stay unchanged.
Hypothesis targets the weak/saturated foot signal at measured10–25cm errors;
actual full-motion improvement remains unverified.

New continuation imports evaluated500 actor, critic, Adam moments and counters
exactly. Simulation/RNG/history initializes afresh; no exact simulator resume
claim. Fresh-only original runner and all previous source pins stay untouched.
Smoke58811 EXIT0:500→502,64 transitions; independent audit verifies all64 rewards,
all Adam steps4000→4016, exact imported learner, all groups change, frozen base
unchanged. Smoke checkpoint4b1d945c9d25b2b8362a859b926372f4e2ae07a06b32594a5afd6e100e0bf630.
Nine focused tests and Ruff E/F pass. Initial CPU loader audit64313 EXIT0:
768 captured parent500 decisions and decoder inputs reproduced bit-exact;
both wrong-checkpoint-family loads correctly rejected.

Main run94600 is ACTIVE under
artifacts/g1_true23_pico_foot_precision_20260910_v1/train500_v1.
It imports original learner500 directly, not the smoke. Fixed500 additional
updates/1,024,000 new transitions, milestones500/600/1000. First update501 is
verified, with exact parent learner preserved. The log's inherited
original_core_920ms_fresh name does not describe actual import semantics; use
parent_import.json and checked snapshot state. Full002/003/008/PICO evaluator
and independent replay auditor are prepared for600/1000. No milestone evaluated
yet. Do not duplicate this live job. Goal ACTIVE; deployment NOT READY.
New experiment:artifacts/g1_true23_pico_foot_precision_20260910_v1/EXPERIMENT.md.

## 2026-09-10: continuous PICO500 complete;69.34s source, full-body deployment still rejected

Run5685 EXIT0:500 uninterrupted outer updates/1,024,000 transitions/4,000
optimizer steps,1333.83s. Checkpoint047b7387f8268d2ff8032c8b74764de6a3f0321f4245814c0f5a909e805dc065.
All original frozen weights unchanged; all seven adapters/Root9/critic/std
changed. Full reward/optimizer audit passes. This is completed training, not
deployment qualification. Do not relaunch it.

Full500 requests complete1417/1417,1569/1569,1114/1114,3817/6530 controls for
002/003/008/PICO. All four full-body tracking screens fail. PICO improves from
30.48s at100 to69.34/115.60s at500 (untouched SONIC50.60s), then unchanged
native joint-range preview rejects. At matched baseline50.60s: root p95
5.239931→0.359780m; arms0.283277→0.266280rad; legs worsen0.163436→0.169004rad.
Full candidate69.34s root p95 is1.053879m, not0.359780m. Walking feet remain
16–22cm relative error. No source crop, retime, target/gain/limit change or
hardware operation. Same100/500 prefix comparison confirms root/arm gains
alongside leg/foot regressions; neither checkpoint is promoted.

Independent final audit96429 EXIT0 verifies79,170 fresh saved-torque2ms steps,
7,917 landmark vectors and384 independent policy reinferences exactly. Training
learning-curve analysis preserves nonstationary population limitations; mean
feet-world RMS first/last100 is0.133619/0.141944m, not improving. Current normal
runner is fresh-only; preserving trained learner state requires an explicitly
validated new continuation, not another fresh restart or claimed exact RNG/
simulation resume. No automatic training extension is running. Goal ACTIVE.
Detailed result/reproducible drivers:artifacts/g1_true23_pico_training_20260910_v1/OUTCOME.md.
Offline rendering91010 EXIT0:3818 frames/50fps/76.36s encoded, including7s entry
and69.34s failed source. Video hash9b6246b81c05837e87d5786a67a0b49edb5f460e46994eb9331066734c4afec3;
15s/75s frames inspected, current Codex panel open requested. Final right ankle
roll0.259895739rad, upper limit0.2618, outward speed3.870868641rad/s: immediate
range risk remains, not a reason to bypass the guard. Renderer executes no new
policy/dynamics. Scoped16 tests pass35.26s; Ruff E/F and diff checks pass. All
jobs now terminal; goal remains ACTIVE and unresolved. Earlier active-training
entries below are historical.

## 2026-09-10: PICO100 full-motion replay rejected; continuous500 run continues

New-bank100 checkpoint evaluated on complete fixed002/003/008/PICO requests.
Actual/requested controls:1417/1417,1105/1569,1114/1114,1874/6530.
All four fail unchanged full-body tracking. PICO completes1524/5780 source
controls (30.48/115.60s), then the unchanged next-step joint-range guard rejects.
Its source leg RMSE0.143495rad, root p950.678713m, relative feet12.32/13.55cm.
Do not mistake range-excess0 on the executed prefix for full physical acceptance.

Equal100-update comparison against old PICO-excluded training also rejects this
milestone: old PICO reaches37.34s; same30.48s prefix old/new legs0.136782/0.143495,
root0.562364/0.678713m. Root improvement against untouched SONIC does not establish
improvement against the earlier trained candidate. No controller promoted.

Independent audit84851 EXIT0:55,100 saved-torque2ms steps and5,510 landmark
vectors reproduced bit-exact;384 separate raw-action/decoder-input reinferences
exact. Summary hash5caf650acd27296c113632034fc6c4e3dd051a44398dddc05881e67e904ab7e9.
Evidence:artifacts/g1_true23_pico_training_20260910_v1/eval100_v1, including
independent_audit.json and equal_budget_comparison.json. Consistent evidence is
not successful tracking. Fixed continuous500 run5685 remains ACTIVE, verified
through400 updates/819,200 transitions, without actor/optimizer resets. Finish
its predeclared500 checkpoint comparison before deciding the branch outcome.
No automatic training extension, relaxed limit, physical command or readiness
claim. Goal ACTIVE. The earlier first-update-only status below is historical.

## 2026-09-10: existing-PICO bank and GPU smoke pass; continuous500-update run active

New bank includes only saved walk002/003 and PICO-FreeDancing, excluding008
from optimizer and unrelated DadDance entirely.7,266 source frames/9,549
lifecycle frames preserve all original native channels and original29 task
geometry bit-exact. Bank hash1ed51fb48efa5ee6ec2e8c32d0fe58d5455ac1b77ce001abb1f034b65df09291.
Seven data-ownership/telemetry tests pass. Trainer truthfully records
pico_used_for_training=true, versus false in the earlier normal-core campaign.

GPU smoke90333 EXIT0:2updates,4envs,8controls =64 actual transitions;
all trainable groups change, frozen base unchanged, initial mean exact.
Checkpoint a02918e09e3996964c4f65f95116588e3001dc2e79b3aaf79ef35c5b40e2248a.
Independent arithmetic audit passes all64 rewards, all7LoRA A/B matrices,
Root9/exploration/critic change, and gradient bound0.500000025. Actual sampled
exposure includes16PICO and48walk003 env-controls; no walk002 smoke exposure
claimed. This proves wiring, not a learned full-body motion.

Fresh continuous500-update128env16step run5685 is ACTIVE under
artifacts/g1_true23_pico_training_20260910_v1/train500_v1. It has completed its
first PPO update; planned1,024,000transitions, checkpoints0/100/250/500, no
inter-milestone actor/optimizer resets. Full fixed CPU evaluator prepared at
evaluate_checkpoint.py;100 and500 checks must include complete002/003/008/PICO,
standing entry/return and unchanged physical/fidelity scoring. No milestone
has yet passed full-motion evaluation. Do not relaunch this active run.

Full source/derived references, physics, reward, original source scales and
native limits unchanged. Input telemetry sampled by explicit control index;
full reward/gradient audit retained. Goal ACTIVE, hardware NOT READY. No live
robot command. Older native124 rejection and runtime results remain below.

## 2026-09-10: native124 comparison rejected; correcting PICO training coverage

Goal ACTIVE; not deployment-ready. The fixed eight-case selected native124
matrix fails both q9 and current-q10 reference boundaries on all four existing
recordings. Six walking cases fall; PICO raw-bound failures at18.52/18.34s also
have velocity ratios1.03346/1.08238. All55,020 saved-torque physics substeps
reintegrate bit-exact;5,504 observations and every ONNX action independently
checked. Native124 has worse leg/relative-foot tracking on matched PICO prefixes
than unchanged SONIC. No teacher labels or controller swap adopted. Nine new
tests pass. Setup/report-writing failures preserved, including first saved
walking trace recovered without rerunning it. Full details:
artifacts/g1_true23_native124_pico_20260910_v1/OUTCOME.md.

Next concrete change is dataset ownership, not another virtual-joint model:
prior normal-core100-update training explicitly excludes PICO. Prepare only
existing walk002/003/PICO full lifecycles, keeping008 outside optimizer and
removing unrelated DadDance. Preserve original29 intent and all native physical
gates. New bank/wiring smoke and bounded continuous training are planned in
artifacts/g1_true23_pico_training_20260910_v1/EXPERIMENT.md; not yet completed.
No physical commands or hardware mode changes performed.

## 2026-09-10: prepared-reference/non-spinning runtime evaluated;5/6 pass, one18.67ms late wake still fails

Runtime optimization is fully evaluated, not a new tracking policy. Seven
reference-FK tests pass; all1817 existing source packets have identical fields.
All656 old and prepared virtual walking states match prior measured qpos/qvel/
time exactly. Legacy reference p95~1.9ms drops~0.7ms in isolated comparison.

Fixed prepared-FK default-spin matrix63857:3/6 scenario checks pass; nominal
walk SONIC controls19/656/47 of656, pause150/200 before timing stop. Gap and
payload correctly stop source at200 and latch balance. Full success was not
selected from these repeats. Added no-spin CPU profile for all three ONNX
sessions, preserving defaults otherwise. All656 saved inference pairs and full
656-source+250-balance virtual native physics are bit-exact against baseline.

Fixed no-spin matrix14480:5/6 scenario checks pass; nominal656/212/656, all three
input-fault cases200SONIC+706balance. Remaining failure atcontrol211: wake
lateness18.674451ms plus12.274717ms execution (7.910008ms inference) crosses
next20ms deadline. Maximum execution across no-spin cases14.232350ms, but wake
jitter still disqualifies complete timing. No implicit restart, source cropping,
limit change, hardware action or newly trained controller.

Audit69376 verifies all twelve cases,107/110 source pins each, timing/fault/
delivery arithmetic and unchanged source-policy prefixes. Fresh saved-torque
native integration matches108,720 physics substeps bit-exact qpos/qvel. Range
excess0, effort ratio<=1, velocity ratio0.569275.44 scoped runtime tests pass
before the final added CLI-completion regression; final rerun follows below.
New opt-in run_g1_true23_prepared_paced_sim requires complete source as well as
physical/timing screens for exit0; a fallback tail no longer implies success.
Its helpers/shared runtime were exercised; this is still a SIM diagnostic.

Full walking fidelity is BIT-EXACT to the failed baseline: legs0.207371rad,
relative feet26.61/20.83cm, root p951.05642m. Optimizations improve runtime cost,
not motion accuracy. Goal ACTIVE; physical deployment NOT READY. All timed/
preflight/audit jobs terminal. Evidence and setup errors preserved under
artifacts/g1_true23_reference_runtime_20260910_v1/OUTCOME.md.

Final verification39936 EXIT0:45 scoped runtime tests pass in15.00s, including
the full-source CLI-success regression; new CLI --help imports successfully.
Scoped new-module/driver/auditor Ruff E/F and document diff checks pass. No
simulation, training, audit or robot process remains running from this turn.

## 2026-09-10: current-observer experiment rejected; reference runtime optimization under verification

Goal ACTIVE. Current-before-inference momentum observer is now implemented and
fully audited, superseding the planned-next-step paragraph below. Four full
source-only preflights remain exact;13 focused tests pass. The separately copied
measured-history benchmark passes exact old/new no-op control comparison and
copy-isolation tests; executed older benchmark sources remain unchanged.

Native trial46124 finishes2002/6530 controls,1652/5780 source=33.04/115.60s,
then rejects an internal missing-axis range crossing0.01790235rad. Native
velocity ratio0.755723; independent saved-target replay of20,020 substeps is
bit-exact. Audit82663 EXIT0 reconstructs2003 histories/predictions/assimilation
updates and96 independent frozen-source reinferences exactly.

On IDENTICAL25.58s source prefix, current-observer versus after-inference
momentum: root p950.310110 versus0.218382m; leg RMSE0.120386 versus0.119676rad;
relative ankle p950.099711/0.117093 versus0.098996/0.115631m. New current-observer
variant is rejected. Entire available33.04s source has root p950.758982m and
leg RMSE0.145735rad. Neither variant completes the source/return or qualifies
full-body teleop. No further virtual-model tuning justified by this result.
Details: artifacts/g1_true23_current_observer_20260910_v1/OUTCOME.md.

Rechecked original23dofsonic task: its "ready" result reports integration and
input delivery, not the independently measured leg/foot fidelity required here.
Existing native training/reference results do not supply a hidden qualified
controller. No new training, imported checkpoint, physical robot command or
changed gain/effort/range/acceptance threshold this continuation.

Next implemented runtime correction: reference-only FK formerly allocates
MjData and runs two full mj_forward calls per packet. New opt-in
gear_sonic/teleop/kinematic_reference.py prepares separate scratch states and
uses mj_kinematics only for reference transforms. Native plant untouched.
All1817 existing002/003/008 packets have identical fields. Reference-conversion
p95 legacy/prepared is1.900/0.708,1.874/0.701,1.770/0.637ms. Seven focused tests
pass after correcting the test fixture to use the existing packet-file loader;
the first six fixture KeyErrors occurred before the preflight started. Full
virtual baseline comparison and fixed six-case actual-clock matrix pending.
This saves runtime work; it does not yet prove the earlier25.336ms outlier's
cause or solve tracking. Evidence: artifacts/g1_true23_reference_runtime_20260910_v1/.

## 2026-09-10: new deployment goal created; momentum update passes former stop, ankle guard still stops full source

Latest user requests fixing the stall and making deployment a goal. get_goal
returned no active goal; create_goal succeeded at1789022140. Goal now ACTIVE:
full-body PICO on actual true23 using existing clips, native SIM tracking and
lifecycle/timing first, staged live/hardware validation later. No robot commands.

New evidence: artifacts/g1_true23_momentum_update_20260910_v1/.
Read-only diagnostic14701 reproduces all1630 old internal predictions exactly.
Copying native root/retained state while keeping missing velocities introduces
a generalized momentum mismatch: equivalent virtual waist-roll correction p95
1.29845rad/s, max5.60854rad/s; matched source correction exactly0. This justifies
a model correction experiment, not a proven complete instability explanation.

Added g1_true23_momentum_source_model.py: update internal missing velocities by
M_new_mm*v_new_m=(M_old*v_old)_m-M_new_mo*v_measured_o. No missing position clamp,
native state write, gain/cap change, encoder/source change or trained tensor.
Full source-only preflight14335 passes6530/1417/1569/1114 controls exactly,
with zero corrections when source/measurements match.20 focused tests pass.

One native trial77565 is REJECTED, but advances beyond the former25.58s stop:
1980/6530 controls,1630/5780 source controls=32.60/115.60s. Missing virtual joint
range excess0 through this prefix. At next control1980 the unchanged native
inward range search rejects. Actual native range/effort excess0; velocity ratio
0.5715422. Independent saved-target integration matches all19,800 native
substeps exactly. Audit43210 reconstructs1981 histories/1980 forecasts and all
momentum updates;96 independent frozen-source reinferences are bit-exact.

Same25.58s source prefix, new momentum / prior virtual copy / zero-model:
leg RMSE0.119676/0.124231/0.124182rad; root p950.218382/0.228086/0.813480m;
arm RMSE0.263045/0.261937/0.295870rad. New left/right relative foot p95
0.098996/0.115631m still fails fidelity. Over its longer32.60s prefix, root
p95 grows0.504244m and leg RMSE0.137673rad. Do not compare unequal horizons
as an improvement or claim full-source/return/timing/hardware readiness.

Stop diagnosis reproduces all8 failed inward candidates: right ankle roll is
0.233984rad, +2.56658rad/s,27.816mrad from its0.2618rad upper bound.100ms
constant-target probes flag risk at1978/1979 while20ms previews still pass.
These are counterfactual probes, not predictions of actual future policy.

Added OFFLINE g1_true23_coordinated_range_preview.py to test search incompleteness.
At the one saved rejected state, coordinated right hip-pitch/ankle-pitch/roll
targets satisfy all23 ranges across10 substeps, where ankle-only search fails.
However448 predictions take500.5ms and change targets up to2.38743rad. A checked
12-step segment refinement only reduces this to2.15544rad. No coordinated
closed-loop trial, runtime promotion or slew/real-time claim. It is not an
acceptable live fix. Combined40780 EXIT0:28 tests passed32.90s; scoped Ruff PASS.

Next concrete hypothesis: the current momentum observer updates AFTER policy
inference, so policy sees pre-update virtual velocity alongside current actual
native measurements. Move this explicit assimilation before policy inference
using copied measurement inputs, preserving old executed sources and native
physics/guards. Not yet implemented/tested at this log entry. No blind rerun,
training sweep, virtual reset or safety-limit relaxation.

## 2026-09-10: internal source-state experiment improves root/arms on a common prefix; full-body trial REJECTED

New work stays in this separate repository; no physical transport, robot action,
training job, gain/limit change or controller promotion. Evidence root:
artifacts/g1_true23_virtual_state_20260910_v1/OUTCOME.md.

First fitted a causal affine missing-axis model on the complete existing source29
PICO trace. Stable recurrence, but all three fit-excluded walking checks worsened
missing velocity and retained leg-command consistency. calibration_v1 is REJECTED;
no native rollout used it. The initial loader setup failure is preserved separately.
Walking recordings were previously used in development, not pristine holdouts.

Distinct physics hypothesis then used the pinned source29 MuJoCo model as an
INTERNAL predictor: current measured root/retained23 q/dq are copied into that
separate model, while six missing axes persist as explicitly hypothetical states.
Only those six q/dq/action history channels change; physical native23 state,
gains, limits, source timing/references and existing target/range guards stay
unchanged. No extra physical joints or sensors are represented as existing.
Source-only effort settings remain separate from actual native motor ratings.

Prediction preflight31855 EXIT0: all6530 PICO,1417/1569/1114 walking controls match
the source simulator's missing q/dq EXACTLY with no missing-state truth input,
reset or native dynamics. Existing source29 tracking and ankle-range failures
remain failures; this preflight verifies prediction code, not a qualified teacher.
New modules g1_true23_virtual_source_model.py / g1_true23_virtual_source_history.py
remain SIM-only hypotheses. Focused5187 EXIT0:26 tests passed15.08s, including
history ordering/isolation and terminal preview/prediction/range failure; Ruff E/F PASS.

One native23 PICO trial49346 EXIT0 (evidence writer succeeded; CONTROLLER FAILED):
1629/6530 physical controls,1279/5780 source controls =25.58/115.60s. Internal
waist_roll_joint prediction crossed its0.52rad bound to0.525826856; candidate
rejected before the next physical control. Native measured hard-range and
commanded/engine effort excesses are0; maximum native velocity ratio0.3479304.
Both initial standing/ramp durations integrated, but source/return incomplete;
not normal-standing acquisition, live fallback, real-time or hardware handback proof.

Identical25.58s source prefix, internal model vs existing zero-model baseline:
root position p950.228086 vs0.813480m; arm joint RMSE0.261937 vs0.295870rad;
leg RMSE0.124231 vs0.124182rad (essentially unchanged, slightly worse).
Pelvis-centered ankle-origin p95 L/R0.097457/0.123552 vs0.117022/0.131525m.
World ankle p950.225429/0.231147m still fails unchanged0.05m screens. Original
pelvis-relative hand-point errors slightly worsen despite aggregate arm-joint
improvement. These are prefix-only gains, not full-body tracking readiness.

Independent saved-target replay matches all16,290 native physics substeps
qpos/qvel bit-exact. Separate audit53644 EXIT0 reconstructs all1630 histories
and internal forecasts;96 independently loaded frozen-source reinferences match
raw29/token64 exactly. Virtual waist-roll oscillation reaches11.4265rad/s;
at rejection it is moving+4.84696rad/s while requested target is-3.89365rad and
source-only actuator is braking at-50Nm. Naive target clamping is not an identified
fix: at that starting state, clamping to-0.52rad reduces requested braking from
-133.495 to-37.342Nm before clipping. No bounds relaxed, virtual resets, blind
reruns or extra policy training. Goal ACTIVE; model stability AND actual leg/foot
fidelity remain unresolved. Runtime fault-path results below remain unchanged.

Render2235 EXIT0 produces all1630 measured native states: H264960x720/50Hz,
32.60s including initial frame. Frames750/1629 visually inspected. Camera follows
root; not a world-tracking comparison or evidence of a successful full clip.
Video: native_actual_v1/rejected_virtual_model.measured.mp4 under this experiment.
SHA25673012f9cbc807ee3469d313bd4c01c4db249b4ee81e774dc1e978acf2f07ecf7.
All new experiment, focused-test, audit and render jobs are terminal.

## 2026-09-10: actual-clock localhost SIM implemented; three fault cases pass, uninterrupted timing FAILS

Active saved-clip full-body PICO goal continues. Added paced_sim_runtime.py,
run_g1_true23_paced_sim.py and test_g1_true23_paced_saved_stream.py in this
separate repo. No physical transport or robot/mode commands. The new consumer
receives nonblocking, uses a bounded FIFO, checks actual-use packet age and
executes one unchanged20ms controller step per measured wall-clock deadline.
Missing/bad input latches the existing balance actor without freezing physics;
returned packets cannot rearm SONIC. All2ms physics substeps and wall-clock
start/finish/deadline/source timestamps are retained, including balance/failure.
The10physics substeps remain batched inside each50Hz control, not independently
wall-paced. No hard-real-time OS or physical500Hz qualification is claimed.

Actual startup68149 failed before any control: packet age122.045ms exceeded
unchanged100ms. No controller retry was hidden. Independent no-dynamics
profiling16587 measured compiled-model serialization/hash231–291ms. It had run
after subscription; prepare_stream now performs it BEFORE subscription. The
failed executed sources are preserved under end-of-stream-v1/executed_sources.
Startup uses two received frames and40ms buffering; source values/speed remain
unchanged. This is source-pose initialization, not normal-standing acquisition.

Actual full-source attempt61400:906/906 total controls, but only394 SONIC then
512 balance. Control393 took25.336ms and missed its20ms compute budget. Runtime
fault latches the existing timeout activation, with a distinct scheduler cause
in the report. The schedule rebases visibly; no rapid catch-up, timestep change
or relaxed timeout hides the failure. All656 published packets arrived intact.
Final maximum joint speed0.002026rad/s. This trial FAILS uninterrupted execution;
it did not complete the full requested source motion or its expected EOF path.

Three separate actual-stream fault trials99728/45569/21001 all EXIT0:500ms pause,
one omitted packet and malformed JSON each latch at control200 and complete
200 SONIC +706 balance controls. Each runs18.12s actual wall time with zero
compute misses. Maximum execution18.640/15.877/17.588ms; later source packets
remain ignored. These are fault-response passes, not full remaining-motion
tracking, recovery to SONIC or firmware handback.

Independent saved-array audit EXIT0 verifies104 source pins per case, all
36,240 new paced physics substeps, actual state/substep correspondence, packet
hash sequences, timing math and strict latch. All actual joint-range and
commanded-effort excesses0; maximum motor velocity ratio0.569275. Every executed
SONIC prefix is bit-exact to the prior baseline. No tracking improvement.
Evidence: artifacts/g1_true23_vr_paced_20260910_v1/verification.json and OUTCOME.md.
Focused81432 EXIT0:45 tests passed12.87s across paced/clocked sessions and
existing paired-live/transport contracts. Scoped new-source Ruff E/F PASS.
All five stream attempts, both profiling jobs and the independent audit are
terminal; no experiment remains running from this continuation.

Bounded inference profiling30113 adds one656-control virtual capture that
matches the previous full baseline exactly, then compares1968 same-input
inferences. Default/no-spinning/single-thread-no-spinning output and token/
history arrays are all bit-exact; inference p955.600/6.546/8.514ms. Alternatives
were slower and are NOT adopted. Captured Python GC maximum0.0333ms does not
support a GC-stall explanation. This probe does not localize the earlier25ms
outlier or establish real-time qualification. No inference settings changed.

Earlier proposed learned-codec/encoder-blindness explanations were ruled out
by original transfer's analytic codec and existing lower240 token perturbation
evidence. Fresh normal64 critic explained variance0.91704 at update100 also
does not support a broken-critic claim from raw value loss alone. No redundant
policy training launched. Leg/foot/arm fidelity remains unqualified; known
full-clip baseline leg RMSE0.207371rad and foot p950.266102/0.208334m stand.
Goal ACTIVE. No promotion, commit/push, new source download or physical damping
diagnosis. Remaining work includes worst-case control-path timing, actual
tracking, standing acquisition, explicit re-entry and later live/hardware checks.

## 2026-09-10: fixed-rate saved-input fault simulation implemented; six cases pass, motion tracking unchanged

Previous turn created the active goal (administrative progress, no controller
gain). This continuation makes executable SIM progress. Training/codec/timing
and contact-adaptation records were checked; rejected experiments were not rerun.
An uncovered simulation boundary was addressed: the old live consumer blocks
waiting for input and does not advance physics during that wait; its optional
trace also excludes the subsequent transport fallback hold. It cannot by itself
prove continuous-time dropout behavior of moving legs.

Added clocked_sim_session.py and record_g1_true23_clocked_input_sim.py, without
editing old executed sources. One exact virtual20-ms deadline -> one controller
tick / ten2-ms physics steps. Missing, malformed, gap or stale input latches the
existing balance actor immediately; resumed input cannot silently rearm SONIC.
Physics failures cannot retry/reset implicitly. An instance-only forwarding
observer records every commanded torque, joint position/velocity and substep
time without changing the controller's returned actions or global MuJoCo module.
This is a NEW deterministic saved-input path, not an updated real-time ZMQ
consumer or reproduction of the historical500-ms receive-timeout behavior.

Actual paired breadth25 walk002 results (all jobs terminal):
- nominal31281:656 SONIC /0 balance, exact baseline qpos/qvel/time;
- pause62237:200 SONIC /456 balance;500-ms gap begins at4s;
- sequence-gap87000 and payload57960:200 SONIC /456 balance each;
- stale-start95112:0 SONIC /656 balance, stale packet rejected before inference;
- EOF92325:656 SONIC /250 balance, full source plus5-second active balance tail.

All six fault-classification/timing, latch and limited physical screens pass.
These are4,186 new controls /41,860 actual physics substeps, independently
recounted from all traces in verification.json: no measured joint-range excess,
commanded effort <=unchanged caps, maximum motor velocity ratio0.569275 (stale
startup0.348740). No unmeasured contact/slip, command-slew or broader robustness
claim. EOF final max joint speed0.006822rad/s. No torque-off/damping-only command
is substituted for the balance actor. This does not establish normal firmware
mode, physical handback or the cause of earlier motor damping incidents.

Fault cases intentionally stop SONIC; returned packets remain ignored while
balance runs. They are NOT full remaining-motion completion or SONIC re-entry.
Nominal trace equality means known leg RMSE0.207371rad and foot p950.266102/
0.208334m remain unchanged. Tracking, standing acquisition, explicit re-entry,
real receiver timing and physical deployment remain unqualified. Goal ACTIVE.

Focused10360 EXIT0:27 tests passed12.41s (new clocked session, old paired-live
receiver and transport contracts), scoped Ruff E/F PASS. Render38951 EXIT0:
907 states/50Hz/H264/960x720/18.14s; includes initial frame and complete balance
tail. Frames700/906 inspected, ffprobe verified. Measured-video SHA256
a633029929f5f04dfe6b806f48d25ae3f3eb384884f9f4d23d08d36615941f31.
Evidence, commands and remaining boundaries:
artifacts/g1_true23_vr_clocked_20260910_v1/OUTCOME.md.
No new data download, training, physical transport, robot actuation, controller
promotion, gain/limit changes, commit or push. All new files stay in the separate
implementation repository. Next integration must preserve these clock/trace
properties; a stable fallback must not be promoted as full-body tracking.

## 2026-09-10: new full-body PICO recorded-input goal created and ACTIVE

User explicitly requests a new goal: working PICO VR teleop using existing
clips toward deployment. Fresh get_goal returned null; create_goal succeeded
with active status, createdAt1789015112. Earlier duplicate-goal failures are
historical; no unfinished goal was falsely completed to permit creation.

Scope: SONIC on actual native23 (12 legs, waist yaw, 10 arm joints), separate
implementation folder, existing recordings only; no headset prerequisite for
simulation. Demonstrate full recorded-motion tracking, timing, joint/effort
limits, standing entry/return and simulated pause/disconnect/recovery. Deliver
reproducible measured replays and a tested configuration/procedure for later
supervised hardware validation. No upper-body substitute or unrelated dance
campaign. Do not label prior-used clips as untouched validation data. Live input
and physical handback remain separate checks that recordings cannot establish.

This turn creates the goal and updates its documented status; no controller
improvement or new simulation is claimed. Baseline remains the immediately
preceding 656-control public walking test with failed foot/joint tracking.
No physical robot actuation, limit changes, policy promotion, commit or push.

## 2026-09-10: explicit public-motion request resumed; full-body walking played, tracking still fails

User explicitly requests internet PICO motion playback including legs. This
supersedes the immediately preceding headset-only work prerequisite. Saved
motion is authorized for this step; it does not replace later live-input tests.
No robot/transport/mode/motor operation, policy promotion, training, commit or
push. The existing goal remains unfinished; create_goal rejected a replacement
because the old goal is unfinished. No false completion used to bypass that.

Fresh paired breadth25, source-orientation-preserving walk002 replay1188 EXIT0:
656/656 controls, 13.12 s, no fallback, minimum height0.737324m, maximum tilt
0.265008rad. Both six-joint legs remain actuated. Source is pinned TWIST2 robot-
retargeted motion from its PICO-based workflow, not raw headset capture. Source
667 frames at50Hz, existing10-frame warmup plus one tail interval omitted;
no crop, speed change or generated holds. Source-pose initialization, not normal
standing acquisition/return. Trace SHA256
00f1695879d36229c3f02e623cfbb5c57e70ef360d55942166e9e39ba90c7a0a,
exactly the historical paired-orientation result: **no policy improvement**.

Fresh tracking: legs RMSE0.207371rad, arms0.596487rad; pelvis-relative foot p95
left0.266102m/right0.208334m; root p951.056420m. Recorder passed=true means stable
complete integration, not full-body tracking. Deployment remains NOT READY.

New artifact-only render_walk.py produced a fixed-world side-by-side original29
prescribed reference versus measured native23 video. No new dynamics, per-frame
alignment or fitted time lag in rendering. Session95872 EXIT0; ffprobe confirms
657 frames,50Hz,H264,1600x688,13.14s including initial-state frame. Frame400
visually inspected. Video displayed inline in the task. Evidence directory:
C:/Users/camer/sonic23_sim_artifacts/public_vr_legs_20260910_v1/walk002_paired/.
Files: source29_vs_measured23.mp4, render.json, report.json, tracking.json,
measured_trace.npz and four frame previews. Source metadata remains pinned in
internet_pico_20260909_v1/walk002/source_report.json.

New artifact-only diagnose_commands.py/8632 EXIT0 observes raw/decoded targets
without changing policy returns. Exact qpos/qvel/time equality verified against
the fresh baseline. Both knees show sustained bias: reference means0.0825/
0.1103rad, requested targets0.3336/0.3618rad, measured0.3716/0.4323rad. This
excludes disabled legs, not all other causes. Policy targets need dynamic
offsets, so direct target/reference mismatch is not itself a decoding defect.
No controller fix inferred solely from this diagnosis. Earlier target-envelope,
source timing/codec, root-feedback and training failures were checked; their
findings are not relabelled as new. command_diagnostic.json and the hash-bound
observed_commands.npz preserve all23 joints and diagnostic-only lag measures.

VR_READINESS.md now separates the authorized saved-source work from the later
real-headset test. No hardware interlocks or acceptance limits changed. Render
and replay jobs are terminal; no blind parameter sweep launched. Full-body
tracking, live capture, acquisition and physical handback remain unfinished.

## 2026-09-10 03:19 UTC: third consecutive live-input blocker check; goal blocked pending operator input

Previous goal turn = PROGRESS: repaired CRLF receiver launcher, started the
actual XR server, added distinct availability/tracking diagnostics and passed
50 regressions. Current turn is a VERIFIED WAIT on receiver session53393
(still live), plus a fresh bounded input check; not new controller progress.
Linux PID351 still owns RoboticsServiceProcess. SDK connects to its server.
100 snapshots over5 seconds contain no valid headset tracking health/body
stream. ADB remains empty; current Wi-Fi192.168.1.7 is unchanged. Evidence:
vr_priority_20260910_v1/input_status_blocked_audit.json and its hash-bound
input_status_blocked_audit.tracking.json. No input or policy data fabricated.

Same required operator input is missing across three consecutive goal turns:
(1) VR priority redirect/preflight, (2) receiver repair and poststart check,
(3) this fresh receiver-live/input check. Local startup and network configuration
checks have been exhausted for this step. No headset session or operator reply
is available to inspect/calibrate, capture or run through the real VR simulator.
Further unchanged probes, rerunning saved clips or another dance campaign cannot
complete this required live-input test. Goal is BLOCKED, not complete, awaiting
operator connection/calibration of PICO and body trackers.

This does NOT declare missing input the only deployment issue. Existing foot/
turn tracking, ordinary-standing acquisition/return and physical handback remain
unqualified. None is promoted or relabeled. Resume with an actual live PICO
capture/simulation test; preserve all remaining scope and physical interlocks.
Receiver stays available; no robot/DDS/controller/motor commands, new training,
simulation motion, artifact promotion, commit or push. No automatic polling or
parameter-sweep loop is scheduled while waiting for the operator.

## 2026-09-10 03:17 UTC: actual VR receiver startup fault fixed; receiver running, tracking still unavailable

Previous goal turn = PROGRESS in simulator-load/input verification and priority
change, not controller improvement. This turn found a missed local prerequisite:
no RoboticsServiceProcess or TCP60061/63901 listener existed. The earlier claim
"service responds" was incorrect: matching binary hashes and SDK initialization
did not prove that its server was running. Zero SDK health defaults also do not
prove headset power or calibration status. Historical probe outputs preserved.

WSL mirrored networking has192.168.1.7/24; existing scoped Hyper-V PICO rules
match192.168.1.0/24. No portproxy or firewall edits. Receiver startup then failed
on CRLF in run_pico_robotics_service.sh (`pipefail\r`, then a remaining blank CR).
Normalize that one launcher to LF; add explicit .gitattributes LF checkout rule
and two regression tests. No receiver binary/config, policy or robot changes.

Corrected launcher is RUNNING in exec session53393, Linux PID351, revalidated by
live handle and process/listener checks. SDK now reports actual server connect;
local TCP60061 and LAN TCP63901 listen. This proves local receiver startup, not
the headset-to-PC LAN route. Receiver is left available for the requested VR
input test; no simulator/controller/LowCmd/DDS or physical robot process starts.

Added read-only probe_g1_true23_pico_input.py around unchanged pinned tracking
probe. It checks only local TCP60061 first, skips SDK when absent, separates
receiver_unavailable, receiver_reachable_tracking_unavailable and tracking
results, and never authorizes hardware. Actual new report is
receiver_reachable_tracking_unavailable: no valid headset health/body stream.
Evidence vr_priority_20260910_v1/input_status_receiver_started.json SHA256
299eb49c54d1e0e05792c356a71c6258829d086c8bd79cb5efcdaf1149323b87;
its tracking report is separately hash-bound. Earlier no-receiver and poststart
tracking-only reports retained in the same directory.

Initial25 targeted tests passed; expanded receiver/input/fault/XR24/SOMA suite
42132 EXIT0:50 passed39.71s. Focused Ruff and diff checks pass. Initial Ruff
import-order finding was fixed before final verification. Test jobs terminal;
only requested PICO receiver remains live. VR_READINESS.md updated. Live input,
full-body tracking, standing acquisition/return and physical handback remain
unqualified. Need operator to connect/calibrate PICO; no dance campaign resumed,
no gains/limits/physics/controller edits, hardware commands, commit or push.

## 2026-09-10: user redirects priority to full-body VR; dance campaigns paused

Latest instruction: "fuck dance readiness giveme vr readiness". This supersedes
dance qualification as the immediate workstream. Continue actual PICO input,
native23 full-body control, latency, standing acquisition/return and disconnect
recovery. Do not resume dance retargeting or arbitrary-motion sweeps automatically.
No upper-body-only substitution and no physical readiness claim.

Fresh read-only PICO probe in vr_priority_20260910_v1/pico_health.json completed:
60 snapshots, zero trackers/body roles, no calibration/tracking, matched service
and XRT binary hashes. ADB sees no device. Current Wi-Fi address192.168.1.7
(old192.168.1.182 is stale). Asked operator to connect/calibrate XRoboToolkit.
No robot/DDS/mode/actuation operations. See VR_READINESS.md for current scope.

Fresh paired VR simulator-load preflight15302 EXIT0 confirms23 actuators and
valid encoder/decoder reports; zero controls/physics steps. Artifact
vr_priority_20260910_v1/sim_preflight.json. Receiver/fault, health and XR24/SOMA
regression73685 EXIT0:33 passed41.98s. No controller or source mapping changed;
no new successful motion/live headset/hardware claim. All current jobs terminal.
Next real-VR session requires operator headset connection, not another dance
optimizer. Existing turning/foot tracking and standing/handoff gaps also remain;
do not present missing headset as the only deployment blocker. Goal incomplete.

Prior interrupted work remains experimental: l1_braking_alignment_v4 accepted
4.58/115.60 source seconds, then moving correction-box failure; feet still fail.
Its numerical foot probe did not fix36mm error;115 related tests passed.
No controller improvement or promotion; independent V4 audit was not completed
before the user's redirect. Preserve artifacts, do not continue that campaign.

## 2026-09-10 02:43 UTC: causal foot-priority retargeting implemented; two bound bugs repaired; coupled-root dead end proven

Previous goal turn = PROGRESS: complete PICO geometry rejected independent-pose
IK and identified artificial shoulder discontinuity. Current turn implements
continuous_reference_alignment.py with past-reference/velocity state, current
original29 targets, foot-first fitting, protected torso stage and bounded arms.
All existing physical/reference/rate/fidelity limits, source times and original
arrays retained; no extra lookahead. Initial standing/history bit-exact.
No controller replay: all three full requested6541-frame reference trials fail.

V1 run8796 EXIT0 accepts376 lifecycle poses/15source frames(.30s), then source
correction and temporal boxes conflict. Independent audit identifies shoulder
pitch and wrist roll approaching tighter correction limits at4.89/4.975rad/s.
Static braking had ignored that correction boundary. Additive correction-braking
V2 fixes this, run5424 EXIT0 accepts378 poses/.34source seconds. A second defect
then treats the previous pose as invalid inside the newly shifted current box.
Independent audit proves this check rejects a feasible next interval: shoulder
is.00329862rad outside current box, but the complete next-step intersection exists.

New pure source_correction_temporal_bounds validates past against STATIC bounds,
then next reference against current correction/braking/temporal rows. V3 artifact
uses explicit private function binding into pinned parent solver, without global
module mutation. This is NOT production/live integration. No weights, iteration
counts, thresholds or source speed changed. V3 run71677 EXIT0 accepts508 poses/
147source frames(2.94s), then rejects on root/temporal feasibility. Accepted joint
rates now pass:4.975rad/s,79.6rad/s2. Foot maxima35.889/36.465mm still exceed5mm;
numerous foot solves reach24 iterations. Upper positions pass only this prefix.
CPU conversion p95.101669s; not real-time. None is controller improvement.

Separate exact box calculation at rejected frame508 proves minimum possible
next root-rotation L1=.472220267rad, exceeding internal.44775 and unreserved.45.
Thus more optimizer iterations cannot repair this state: earlier choices must
reserve braking room against the combined rotation constraint. Not a proof of
physical G1/dance infeasibility. Existing normal policy still stops50.60/115.60s.

Independent audits: V1 all376 FK/rates and45 bound files; V2 all378 and49; V3
all508 FK/rates/metrics/protection and51. V3 initial aggregate-equality assertion
failed on1.39e-17m independent hand-offset arithmetic; failed code preserved and
same predeclared2e-15 array tolerance applied to aggregate. No reference rerun,
physical/fidelity tolerance change or bit-exact hand metric claim.105 combined
tests21210 pass36.12s, all six new repo files Ruff/format pass. All processes terminal.

Evidence C:/Users/camer/sonic23_sim_artifacts/internet_pico_20260909_v1/
continuous_reference_alignment_v1/OUTCOME.md and sibling correction_braking_alignment_v2,
current_box_alignment_v3. V3 report30d930ac684a92631c3176f85ade1a98f165181934db7e43276f9b5e4d43c481;
audit1fbe34df49b95f4626fd97abb983d7f9598623807c5fa2d2434190d389734eb1.
No hardware/DDS/modes, physics/gain/effort edits, training, default-policy edits,
promotion, export, commit or push. Goal ACTIVE. Live full-body teleop NOT READY.

## 2026-09-10 02:04 UTC: full public PICO frame-local waist compensation improves geometry; rejected on continuity and feet

Previous goal turn = PROGRESS in hypothesis rejection, not controller improvement.
Current turn adds separate SIM-only g1_true23_torso_reference_alignment.py:
actual source FK transfers missing waist motion into reference pelvis; bounded
leg/arm IK targets original world feet/hands. Original29 arrays and every
6541 lifecycle pose/5780 source frames/115.60s remain unchanged as truth/time.
Missing6 are never physical actuators; no robot-state reanchoring. Native torso
structural offset remains measured. Earlier bounded whole-path pelvis fitting
already existed; this is a frame-local mechanism on full public PICO, not a
new claim that pelvis compensation alone solves native23 dynamics.

Prepare81994 EXIT0: source hand p95 improves201.658/209.376mm→.519/.463mm;
head146.039→10.116mm. These are reference FK metrics, NOT policy tracking.
Worst feet32.178/19.985mm exceed5mm. Joint bounds pass, but speed146.960718rad/s,
acceleration9026.108743rad/s2 fail unchanged5/80. At84.02→84.04s right shoulder
yaw switches2.165340→-.773875rad while original stays1.4→1.4rad. Source GMR
itself also reaches21.479629rad/s and1073.975005rad/s2. No valid teacher claim.
Base tilt peaks.767040rad;183 frames exceed existing.5 envelope. Root correction
and angle/rate/joint-change screens also fail. CPU conversion p95.033311s.
Initial history bit-exact; source timing and all physical limits untouched.

Per preregistered experiment gate: no controller replay of rejected mapping,
no same-clip weight sweep. Existing normal native23 baseline still stops after
50.60/115.60 source seconds. This does not repair or diagnose physical damping.
Independent audit24431 EXIT0 imports no converter and reruns no IK: full task
FK/constraint results reproduce,41 bound inputs verified. Virtual torso origin
residual1.570092e-16m is not whole-body feasibility. No contact/support proof.
15 focused tests91764 pass;62 combined30579 pass27.85s. Initial test caught an
incorrect constant torso-offset expectation, corrected to derived geometry;
initial combined command had nonexistent test filename, corrected without a
motion rerun. Scoped Ruff/format and diff checks pass. Processes all terminal.

Artifacts C:/Users/camer/sonic23_sim_artifacts/internet_pico_20260909_v1/
torso_reference_alignment_v1/OUTCOME.md, geometry_report.json, audit.json,
limit_diagnosis.json. Geometry report
be6cd65a9fad69b16b6493a0000887e9ef87e035bb3e3bffb85ab05c35c068be;
audit66035139a454fee0ba0cde53c61f68fa23e06063a597d019317bf0bb389e0aab.
No hardware/DDS/mode, training, default edits, promotion, export, commit or push.
Goal ACTIVE; full-body live teleop NOT READY. Next reference mechanism must
resolve temporal continuity and foot priority, not merely improve isolated poses.

## 2026-09-10 01:41 UTC: discarded six-action memory tested on native23; rejected after earlier PICO guard stop

Previous goal turn = PROGRESS: v1.1 adapter/run/audit closed an upgrade hypothesis,
not a deployment improvement. Current turn implements one distinct SIM-only
runtime hypothesis in g1_true23_discarded_action_memory.py. Retains six discarded
raw outputs as explicit internal action memory; absent measured q/dq stay zero.
Retained23 applied-target history, full normal SONIC tensors/920ms source horizon,
physical model/gains/limits/preview, original PICO source and6530-control request
stay unchanged. This is NOT earlier retained23 raw-history/low-latency experiment.
No fake physical joint states, missing motor commands or default profile edits.

Preflight:64 saved baseline raw/token/decoder comparisons EXACT. Actual first
control and ten physics substeps match baseline exactly. Run89578 EXIT0 requests
full115.60s, stops1096/6530 controls,746source/14.92s at unchanged range preview.
Unapplied1097th output retained. Matched14.92s root p95 worsens.561633→3.537872m,
leg RMSE.118485→.181321rad. Old normal baseline still fails later at50.60s.
Actual hard-range/applied/engine effort excess0. Target step max1.232614rad/20ms
is not hardware-qualified. No retiming, fallback or postinitial state reset.

Independent audit22847 EXIT0: original29 teacher, not new policy/memory helpers;
all1097 full29/token/retained outputs, measured and virtual histories, accepted
commands and terminal rejection reproduce exactly. All10960 persistent physics
steps match qpos/qvel exactly; preview max difference3.053114e-14. Root/leg and
matched baseline metrics independently exact; hand/foot/head metrics not separately
recomputed. No force events selected; zero residual placeholders are NOT force-
balance proof. Twelve focused tests26856 pass;41 combined70833 pass in19.18s.
Initial combined command named a nonexistent test file and collected zero tests;
corrected command only, no motion rerun. Scoped new-file Ruff/format checks pass.

Verifier2024 EXIT0 rechecks360 bound files. Artifact directory:
C:/Users/camer/sonic23_sim_artifacts/internet_pico_20260909_v1/discarded_action_memory_v1/
OUTCOME.md and verification.json. Actual report
712430d9dc026c02d097ad8f7512c3d94452904139ac1159fe48e1bc657f7b05;
network audit b98544df6e15e55bcc750511bd5d538ead9e39eafac8f4ee11b52736b93eebd5;
receipt53bb1f2a57670739a91310903cf1f582cdb97e1f1fde3d7e741ea43e4af978bd.
All current experiment/audit/test processes terminal. Reject candidate; no more
same-clip memory sweep. This does not isolate morphology as the sole cause or
establish a learned alignment fix. Upstream draft reread confirms analytic
codec plus decoder LoRA, NOT a learned input codec. No training, hardware/DDS,
mode, export, promotion, commit or push. Goal ACTIVE; full-body teleop NOT READY.

## 2026-09-10 01:16 UTC: official SONIC v1.1 adapter verified; full PICO native23 candidate fails earlier

Implemented separate offline g1_sonic_v11_onnx_reference.py with8 focused tests;
35 combined regressions58756 pass in16.65s. Matching official ONNX pair and
observation config are revision/SHA-bound. Actual encoder1751, decoder994,
token64, output29; verified graph routing and robot-heading-only reference
orientation. Old normal encoder1762 was correct for that older release.
No existing native23 controller, gain, physical torque/range limit, preview
guard, deployed checkpoint or source speed changed. No training or robot/DDS.

Same public PICO female1/20231025_032712,115.60s optical full-body motion:
native23 v1.1 run57064 stops1044/6530 controls,694 sourceframes/13.88s,
absolute height/tilt diagnostic. Root p95=5.029433m, leg RMSE=.219658rad;
actual range/effort excess0. Old normal/native23 remains stronger but still
fails at50.60s. On identical13.88s source interval, old native root=.533872m.

Source43 v1.1 run76469 completes6530 controls/all115.60s,29 real body axes,
default5ms hand scene and official source-only1.5x ankle-pitch KP/KD tuning.
Root p95=1.109574m, leg RMSE=.161591rad; tracking screen FAIL. Actual right
ankle-pitch range excess=.05847474rad at physics11334;113 body-range excess
substeps. Applied/engine effort excess0. Source tuning NOT copied to native23.
Matched13.88s source43 root=.261042m versus old source43=.147423m.

Preflight21456:64 independent ONNX/PyTorch comparisons, tokens EXACT,
raw max error1.907349e-6; all64 C++ heading values EXACT. Native audit2727:
1044 decisions/10440 persistent physics steps EXACT; root/leg independently
recomputed, native hand/foot/head metrics not separately recomputed.
Source audit56349:6530 decisions/26120 physics and ALL task metrics EXACT.
First source audit93659 EXIT1 preserved: optimized C++ quaternion arithmetic
differs near zero from standard Hamilton arithmetic. Separate v2 checks C++
at predeclared1e-6 tolerance; maximum2.671007e-16 on298/6530 controls. Source
C++ headings are NOT bit-exact; independently reconstructed network inputs,
outputs and physics remain bit-exact. No actual motion rerun or relaxed limit.

Verifier77848 EXIT0 rechecks458 bound files. Artifact directory:
C:/Users/camer/sonic23_sim_artifacts/internet_pico_20260909_v1/sonic_v11_pico_v1/
OUTCOME.md and verification.json. Source audit summary
27647ea11ea952825a9a4b728ca774fbfa78bbe56dd7f57acdfa11e9f8fcf351;
receipt1f576271d480fdf420e71db7b3ae081c6dcb4c5e3b8319bc5a05cf83be3ce22a.
All current processes terminal. This is compatibility validation and candidate
rejection, NOT native23 policy improvement. No export, promotion or commit.
Real PICO dataset has paired HMD/IMU; current test uses optical body reference,
NOT sensor-only inference/live headset. Goal ACTIVE; live full-body teleop
NOT READY. Version upgrade and settings copying are not demonstrated fixes.

## 2026-09-10 00:35 UTC: full PICO retained-contract trials reject short-prefix motor-settings fix

Continuation = PROGRESS in diagnosis, not a native23 policy improvement.
New isolated artifact normal29_retained_contract_full_pico_v1 runs exactly two
full requested PICO trials after correcting the previous source-only raw10
cutoff. Same frozen29/2ms model, source speed/geometry, normal920ms horizon,
checkpoint, initial state and lifecycle. All29 axes remain real. Trial1 applies
actual native23 retained target bounds/applied-action history; trial2 additionally
maps recorded native23 KP/KD/effort onto ONLY retained23. Missing6 retain source
gains/upstream simulator caps/finite affine targets. Native23 raw bound, physical
ratings, model and actual preview guard remain unchanged. No other four-motion
repeats, training, policy edits, robot/DDS/modes, export, promotion or commit.

Actual5312 EXIT0 records both outcomes: trial1 completes6530/6530 controls and
115.60s source, but full-source root p95=3.175291m, leg RMSE=.119725rad;
range excess=.03686454rad on61 substeps (left ankle pitch minimum-.90953454
versus-.87267). Trial2 fails at2220/6530 controls,1870source/37.40s, absolute
height/tilt diagnostic: minimum rootheight=.11194299m, maximum tilt=1.771706rad.
Root p95=2.940438m, leg RMSE=.292408rad; range excess=.10278819rad on2277
substeps, maximum right shoulder roll-2.35428819 versus-2.2515. Both tracking
screens FAIL; applied/engine effort excess0. Full target range excess is also
reported, including source-only axes; none is a physical hardware command.

Matched first37.40 source seconds, root p95/leg RMSE:
original29 .378065m/.114625rad; bounds+history1.394238/.121359;
bounds+history+native motor arrays2.940438/.292408; actual native23
1.812547/.152223. Native motor variant looks good at20–30s (root p95=.252124m)
then fails badly at30–37.4 (3.105777m). Therefore prior good25.94s prefix is
NOT a full-clip motor-settings fix. On50.60s shared baseline/bounds/native23
interval, root p95=.593318/1.884702/5.239931m. Native23 still has its unchanged
50.60s ankle-preview stop. Clipping mainly hits ankle pitch/roll and hips;
first projected control355. This does not isolate missing axes as sole cause.

Independent audit61668 EXIT0: all8750 network attempts/accepted actions and
87500 physics substeps EXACT, including second failure. Independent command
formula and separately accumulated applied history; new runner/lifecycle/retained
helpers not called. ALL foot/hand/head/root/joint metrics and matched reference
comparisons exact. No dynamics-state resets. Six focused tests pass;42 combined
regressions48703 pass in23.23s. Scoped Ruff and five-file format checks pass.
Read-only diagnostics33731 EXIT0,0 integration: first reporting attempt had
Path/string checksum type error before writing output; fixed without motion rerun.
Final verifier96297 EXIT0 revalidates210 bound files and emits receipt.

Artifacts: C:/Users/camer/sonic23_sim_artifacts/internet_pico_20260909_v1/
normal29_retained_contract_full_pico_v1/OUTCOME.md and verification.json.
Actual summary b5dc3e57d287a9769aad87f303f5b198559f1a00727dafa00618bf7d19bb7726;
audit91aaca62983a342aa6518d06c14b18508b5538713e951458161b55c4f995b383;
receipt e4754e420b7ea12e393d67722bb2c2234059d531a09730295e3a629217cc16d1.
Public PICO provenance rechecked: paired real HMD/IMU + optical body reference;
these tests use optical full-body motion, NOT sensor-only/live inference.
No simple joint deletion/settings-copy solution demonstrated. A new native23
controller/reference mechanism must pass full lifecycle within actual limits;
do not recycle good prefixes or failed short training recipes as readiness.
Goal ACTIVE; full-body live teleop NOT READY. All current processes terminal.

## 2026-09-10 00:01 UTC: source29 raw-cutoff mismatch corrected in isolated artifact; full PICO now integrates, tracking still fails

Meaningful benchmark correction, not native23 deployment progress. Upstream
CreatePolicyCommand and PolicyEngine::Infer read back float32 outputs and apply
the body affine formula without the old offline raw10 cutoff. New separately
declared normal29_upstream_action_v1 artifact reproduces that finite formula;
does not edit old stopped runs or any native23/hardware guard. Explicit native23
model rejection, nonfinite input/target checks and actual model/torque limits
remain. Five focused tests pass. Initial external-path pytest discovery hit
Windows DumpStack.log.tmp; scoped root/confcutdir run fixes collection only.

Two actual full PICO runs19319 EXIT0: frozen29/2ms and upstream43/5ms, both with
same source-YAML torque caps as predecessors. Both complete6530/6530 controls,
including5780/5780 source115.60s and return-reference phases. All old executed
prefix arrays AND formerly rejected action match exactly before continuation.
Return-reference integration is not Unitree mode-handoff or standing qualification.

Both world tracking screens still FAIL. Full-source root p95 is1.184333m and
1.482296m; leg RMSE.115721/.117833rad. Body hard-range excess.01684395rad
(44substeps) and.00312861rad (2substeps); source43 passive fingers reach.00639058.
Applied/engine effort excess0. On unchanged native23's2530-source/50.60s common
interval: native root5.239931m/leg.163436rad; fixed29.593318/.113331;
source43.773379/.116392. Different morphology/gains/bounds mean not a pure
missing-axis causal comparison. Native23 trace/ankle stop remains unchanged.

Independent full audit93467 EXIT0: both complete13060network/91420physics and
all foot/hand/head/root/leg metrics EXACT. Fixed29 ankle-roll excess reproduced
atsubstep51202(q=-.27864395 vs-.2618); source43 ankle-pitch excess at14980
(q=-.87579861 vs-.87267). Source43 minimum rootheight.400684m/max tilt.587033rad;
these are measured quantities, not standing/hardware passes.
Audit b773274a1d7be672d44566512e86d821359a6c75e8a3876e43ef9116c55c250c.
Actual summary
d7098503981c51a4a415d310541e6715d888801de8ad112b7687450870efc6ec.
No new training, policy promotion, native23 modification, robot/DDS/modes,
physical interlock/limit edit, export or commit. Goal ACTIVE; teleop NOT READY.

00:10UTC: comparison render20807 EXIT0. ffprobe verifies2881 decoded frames,
50Hz,57.62s,H264,1920x592. Frames0/700/2880 inspected; same fixed world camera,
prescribed29/measured source29/measured native23. Ends at actual native ankle
stop, no post-stop recovery invented. Final source29/native23 root errors
.792/5.484m visible. Video normal29_upstream_action_v1/display_v1/
pico_source29_corrected_native23.common-prefix.mp4 SHA
0df364479b5ffccf358cd0c5cdf956d0c96429f1b3c45b192f656dfbc2b88356.
OUTCOME.md and verification.json complete; executed sources/trace hashes
rechecked. All current experiment/audit/render processes terminal; goal incomplete.

## 2026-09-09 23:50 UTC: source29/43 timestep trials verified; PICO raw cutoff is not a joint-limit failure

Continuation = PROGRESS. Added isolated named29-body layout/lifecycle supporting
source29 and hand-equipped source43,2/5ms. Nine new tests;33 combined tests pass
(92227,13.44s), scoped Ruff/whitespace pass. Native23 and old controller files
unchanged. Generalized adapter reproduces full old walk0021417control/14170step
trace/source/metrics exactly before trials; new full-actuator traces also match.

Actual17414 and independent audit36613 exit0. Fifteen trials:21132 attempts,
21129 accepted controls,126774 physics substeps, plus separate calibration.
Fixed29/5ms,source43/2ms,source43/5ms respectively run all five requested clips;
upstream YAML effort, original C++ gains/target mapping, raw history, checkpoint
and source geometry/timing/speed stay fixed. Source43 fingers have real free
dynamics, zero applied command. No band, postinitial state rewrite or fallback.
Twelve lifecycles finish, ALL15 world-landmark screens FAIL. Same-prefix root
p95 for PICO.252034/.332004/.253177m,walk002.454214/.463602/.443544,
0031.233720/1.247722/1.264698,0082.294117/2.541538/2.631070,
dance.708145/.692832/.765027. PICO all1647/6530 total,1297/5780source25.94s.
Dance actualbodyrange excess.00403206/.00407575/.00283929rad; other bodycases0.
Passive source43 finger excess reaches.01002869rad. Not physical native23 data.

Independent named mapping plus fresh compiled recipe/hash checks reproduce
every network/token/target/control/body/finger force and persistent state exactly.
Separate FK checks ALL foot/hand/head/root/leg world/relative/orientation metrics
and entire landmark traces exactly, unlike prior effort auditor's narrower FK
coverage. All failures/ranges preserved. No deployment or native23 improvement.

Important new interpretation: saved rejected wrist raw10.04-10.07 corresponds
to only.748244-.750577rad via source scale.0745008703; actual source wrist limit
is+-1.61443rad and measured position.682-.698rad. Read-only arithmetic33010
exit0, no steps/commands. Upstream CreatePolicyCommand at3334 uses scaled action
without this raw10 diagnostic cutoff. Thus source29 truncation is NOT a physical
joint failure; it does not explain native23's real ankle-preview stop. Tracking
already fails before that truncation. Existing traces/guards are not relabelled.
Next bounded source-side question: separately reproduce upstream action handling
on full PICO, preserving simulator model/effort and reporting actual ranges,
without changing native23 controller or removing its hardware interlocks.

Artifacts: internet_pico_20260909_v1/normal29_source_scene_v1/OUTCOME.md.
Actual b491a3aa4698e85c7481ddfc85e97e91052485b153096e6bd0c651c1ef07ce8e;
audit1e29a9254492d6e3ac5ffa8f1eaf62b3d3e4167436c30cf6ad7ce7b9da1ad2ee.
PICO still optical full-body reference, not sensor-only/live inference.
Goal ACTIVE; full-body teleop NOT READY. No hardware/DDS/mode, training,
physical rating/guard edit, export, promotion, commit or push.

## 2026-09-09 23:32 UTC: released ONNX parity passes; source effort correction still fails tracking

Current goal turn = PROGRESS, not blocked. Original normal release ONNX checked
against pinned checkpoint on80 saved observations spanning all five motions:
all24 affine tensors and80 FSQ tokens EXACT; max raw error2.3841858e-6.
Checkpoint-versus-saved outputs EXACT. Actual encoder ABI1762, decoder994;
separate1751 config comment is not a mismatch. This proves fixed-input CPU
parity, not live C++ observation, DDS or TensorRT closed-loop parity.

Found concrete source-SIM effort mismatch: existing CppParameters explicitly
uses older Python legacy limits, not firmware proof. Actual upstream43 YAML
clips ankle/waist roll-pitch50Nm versus legacy25; hip pitch/roll88 versus139,
already internally88 in frozen29. Added SIM-only name-based mapping plus7
tests.35 combined tests pass (12753,15.63s); no native23 rating/guard changes.

Five complete requested effort-only trials3835 and independent audit94748 exit0.
Same frozen29/2ms plant, source geometry/speed/920ms timing, weights, targets,
gains and raw history.7044 attempted/7043 accepted controls,70430 physics steps;
four lifecycles finish, ALL five world-landmark screens FAIL. Paired root p95
old->new: PICO.325928->.295702m,002.414418->.425689,0031.160550->1.198379,
0082.437234->2.206174,dance.684228->.618778. PICO stops1647/6530 controls,
1297/5780 source (25.94s), raw right-wrist-yaw10.128579. Not native23 ankle cause.
Dance actual waist-roll range excess.00388968rad on42 substeps.008 range excess
now0; remaining three0. Corrected simulation efforts are not physical ratings.

First changed engine forces have exactly equal pre-state/target/requested
torque prefixes; all received sources unchanged. Independent fresh43 YAML
mapping (new wrapper not called), source/history/network and every persistent
physics step reproduce exactly0 error. Root/leg and original-intent hand/head FK
metrics independently match exactly too; foot metrics not independently recomputed
in this audit. No state resets.

Read-only upstream model inspection: frozen29 mass35.112142kg,36/35/29state/
actuation at2ms; default hand-equipped scene36.1652352kg,50/49/43 at5ms.
These scene/timestep differences remain untested, not an established fix.
Original23dofsonic task remains old/inactive; its V14 public-motion comparison
already failed (Sep8 15:28 entry), so no duplicate run or readiness adoption.

Artifacts: internet_pico_20260909_v1/normal29_deployment_parity_v1/OUTCOME.md
and normal29_source_effort_v1/OUTCOME.md. Both include verification.json.
Effort summary d5abc5a08d48229fcaaf4c8778178a064280220b14539ff12fdfa76a0b4232c2;
audit e5229a94f597d529b199d7f0798135036572bc2b16edf1656ed620e28756ada6.
No native23 policy improvement, training, hardware/DDS/modes, export or commit.
PICO uses paired optical full-body reference, not sensor-only inference.
Goal ACTIVE. Full-body teleop NOT READY; physical damping cause unproven.

## 2026-09-09 22:58 UTC: 25 target/history/gain comparisons executed; no global fix

This goal turn = PROGRESS, not blocked. Previous normal29 baseline evidence
revalidated from saved files. Added SIM-only retained23 target/history ablation
and 11 focused tests; 28 combined tests pass (final rerun63026: 12.77s). No
native23 policy/controller improvement, training or promotion occurred.

The fixed29 2x2 experiment changes only retained23 bounded versus roundtrip-only
targets and raw versus applied previous-action feedback. Full requested PICO
calibration through its wrapper reproduces every original baseline trace/source
array and early-stop result exactly: 1647 controls,16470 physics substeps.
Actual experiment63435 and independent audit60827 exit0:20 trials,27598 attempted
controls,27593 accepted,275930 persistent2ms physics substeps.15 lifecycles finish,
but ALL20 world-landmark screens fail. Same PICO1297-source interval root p95:
roundtrip/raw0.325928m, bounded/raw0.275113m, roundtrip/applied0.325928m,
bounded/applied0.276716m; untouched native23 remains0.817269m. All four stop
1647/6530 total controls on original29 raw right-wrist-yaw>=10, an axis absent
on native23; not its ankle-stop cause. No PICO return is reached.

Bounding worsens walking002/003/dance; applied history is motion-dependent.
Walking008 bounded/applied stops535/1114 controls and exceeds actual waist-pitch
range0.111915rad, with ankle/waist excursions too. Fair comparison uses only its
185-source common interval, not longer controls from surviving variants.
First projected commands have identical states/inputs/raw/tokens before target
intervention; first affected controls PICO355,002319,003312,008302,dance451.
Roundtrip-only numeric effects are not universally negligible on walking008.

Separate gain/effort-only fixed29 run3391 and audit80510 exit0:5 trials,6528
attempted,6526 accepted,65260 physics substeps. Only retained23 KP/KD/external
effort arrays change to recorded native23 values. Same-prefix PICO root p95
improves0.276716->0.171919m, while walking003 worsens1.384643->2.476488m,
close to native23's2.483157m. Dance worsens0.811933->1.275609m. Native gain swap
therefore does not explain/solve both.3 lifecycles finish; ALL5 tracking screens
fail. PICO stops1648/6530 on raw wrist;008 stops596/1114 with actual right-ankle-
pitch excess0.128210rad. No native23 limit/guard change or hardware command.

Combined25 trials:34126 attempts,34119 accepted controls,341190 independent
physics replay steps, plus separate calibration. Independent target/history
formulas and gain mapping reproduce all compared input/output/token/target/state/
torque arrays exactly0 numerical difference, with no intermediate state writes.
Rejected actions and actual ranges reproduce. Root/leg arithmetic is separately
checked; current audit does not independently recompute every hand/head FK term.
Replay equality and finite output are NOT tracking or deployment passes.

Fresh read-only frozen29 plant inspection records ctrl limits and armature:
hip pitch/roll internally capped88Nm even where C++ external limits139Nm;
all hinge armatures0.01. Gain/external-clip change is not complete native motor,
morphology or source-training physics parity. Neither frozen model nor C++
capture is changed. Earlier native23 exact-C++ gains and inverse-dynamics runs
already failed; don't repeat them as an untried fix or extend rejected LoRA blindly.
Remaining PICO model/missing-axis/coordination gap is not uniquely isolated.
Original29 benchmark itself remains unqualified; establish source-faithful
successful control before treating it as a teacher for further adaptation.

Artifacts under internet_pico_20260909_v1:
normal29_target_factorial_v1/OUTCOME.md and normal29_native_motor_v1/OUTCOME.md.
Factorial summary e76cea59e7247e32aa1dce0ccdf67e63a6017f047523e0e1cba46231f0178cd1;
audit84add8986eba732bd5c1f281ec10a76f40b4fbf23a73ae088ce216ba257d2982.
Gain summary f958087bdec9d30b7529a5352349ae6674ad3654d643d2e44d077668a897bb48;
audit18ff509be482840a692d907c1160dc1abbfc7a23f9bf6413f09eecf99243ef8c.
Executed source/artifact hashes rechecked unchanged. Scoped Ruff passes.
PICO remains paired optical full-body reference, not sensor-only headset
inference. Native23 baseline50.60/115.60source seconds still drifts before its
ankle guard rejects. Full-body teleop NOT READY; physical damping cause unknown.
Goal ACTIVE. No robot/DDS/modes, export, physical rating change or commit/push.

## 2026-09-09 22:18 UTC: matched normal29 PICO baseline executed; command boundary narrowed

This goal turn = PROGRESS, not blocked. Added separate normal29 teleop267/FSQ/
seven-affine decoder and47-sample/920ms lifecycle.80 saved-input preflight
observations exactly match original29 outputs/tokens and selected23 outputs;
17 focused tests pass. All five initial native23-versus29 observations and
retained raw outputs are exactly equal. Old low-latency/G1-encoder evidence
unchanged; no new training or promotion.

Actual full5-case run6509 exited0.29 walking002/003/008/dance complete their
1417/1569/1114/1296 controls, but ALL world-landmark screens fail. Root p95:
0.414418/1.160550/2.437234/0.684228 m versus untouched normal23 on identical
prefixes0.583985/2.483157/2.346464/0.906232 m.29 walking008 exceeds ankle-roll
range0.012264 rad over27substeps; dance ankle-pitch0.004321 rad over30substeps.
Integration completion is not a limits pass.29 gains/physics/filtering differ
from native23; not a morphology-only ablation.

PICO29 stops1647/6530 lifecycle controls,1297/5780 source=25.94s. Rejected raw
right_wrist_yaw_joint10.077461 exceeds unchanged diagnostic |raw|<10. This axis
is ABSENT on native23: it does not explain native23's separate ankle stop.
Same1297-source prefix root p95 normal29/23=0.325928/0.817269 m; leg RMSE
0.109529/0.124425 rad. PICO29 actual range/applied/engine effort excess0.
Return not reached; optical-derived PICO reference, NOT sensor-only/live input.

Independent audit4802 exited0: all7044 attempted network inputs/outputs and
7043accepted targets reproduce exactly; all70430 persistent2ms physics steps
reintegrate qpos/qvel/torques exactly0 numerical difference without intermediate
state writes. Original-intent FK metrics exact0. Current legacy29 right-hand
offset is already+0.025m, equal to original intent; pre-run wording must not be
read as an actual current offset discrepancy.

New read-only boundary_audit.py exits0 on2881 untouched-native23 PICO attempts.
Selected original29 default angles/scales match native source codec EXACTLY.
Target bounding changes1331attempts/1330accepted controls, first control359
(both ankle pitch), maximum accepted target change1.527235rad. Additional range
preview changes ZERO of2880accepted targets, exactly0 difference; it rejects
control2880. Removing final stop cannot fix earlier drift. Recorded hip-pitch
kp40.179238vs99.098427, kd2.557890vs6.308802, effort88vs139Nm differ native23/29;
ankle effort35vs25Nm. No permission to raise physical ratings. Attribution of
trajectory failure to projection versus motor/model dynamics remains unproved.

Next controlled gap: keep original29 model/gains/weights/source fixed and apply
native retained-joint target bounds, separating previous-action semantics.
Do not repeat identical LoRA updates or remove interlocks. Older inverse-
dynamics/feedforward experiments already exist; this is not an untried magic fix.

Artifacts: internet_pico_20260909_v1/original29_normal_v1/OUTCOME.md.
Full summary566f1472aff2ba8ff998074403a5065c9e03677a5e7b17eb12e23d9351e84235;
audit363d35155b4b09bd474bdb9d41e811829f18680df84709957b01c58c8dd5dc57;
boundary54e6e85911a95c40d223084530b39546f96ab2e49d67086e9dcbad104e131101.
Render34794 exits0; ffprobe verifies1648frames/50fps/32.96s/1920x592H264.
Frames0/700/1647 inspected: prescribed29 versus measured29 versus measured23,
fixed world camera, common prefix only; final root errors0.287/0.855m visible.
Video display_v1/pico_source_normal29_native23.common-prefix.mp4,
SHAfc6088b36f2385873b0a6bed6397b4865375590506f6df6815d29a52ab0a36e0.
Focused tests rerun97788:17passed/10.30s; scoped Ruff and diff checks pass.
verification.json records tool completions; comparison video panel queued.
Live full-body teleop NOT READY; goal ACTIVE. Physical damping cause unknown.
No robot/DDS/mode commands, limits changes, export or commit/push.

## 2026-09-09 21:38 UTC: rollout64 fully evaluated; PICO15.42s regression, checkpoint REJECTED

Pipeline46776 completed all stages exit0: fresh100updates/204800transitions/
800optimizersteps, independent reward audit, exact actual PPO-return audit,
five full requested CPU trials and independent network reinference. Training
loop15m42s. Initial actor and critic both bit-exact to normal16; all seven LoRA
A/B, Root9, exploration and critic change, frozen core stays exact.1257training
terminations,81timeouts,1010world-root failures.35focused tests5907 passed.
Checkpoint8619cb9c7122b406322388950880f441c2b452b5dace6ca9b1790810dbbb0f6b.

Four walking/dance lifecycles complete, but ALL original world tracking screens
FAIL. Root p95 normal64:0020.732035m;0032.528176m;0082.188927m;dance0.735636m.
Only008 improves against both comparison policies; dance beats untouched normal
but not normal16. No source trimming, retiming, limit relaxation or pose resets.

PICO stops1121/6530controls,771/5780sourcecontrols=15.42s versus normal1637.34s
and untouched normal50.60s. SAME771source prefix root p95: normal642.031355m,
normal160.264696m, untouched0.590543m. Leg RMSE0.159839/0.115791/0.119217rad.
Right ankle pitch-0.859168rad at-2.739874rad/s: all8bounded candidates predict
crossing hard lower-0.87267rad (nominal min-0.892295). Copied-state analysis adds
80probe substeps, no new policy trial. Does not establish historical hardware
damping cause or explain prior drift. Return/standing phases not reached.

All6518attempted network outputs and6517accepted controls exactly reproduce;
all65170saved-target physics substeps reintegrate qpos/qvel exactly0. Actual
hard-range/applied-effort excess0. This run passes unchanged2e-10 preview check,
max1.115e-13; older normal16 preview failure retained. Target jumps still1.1341rad/
20ms, unqualified for hardware. Empty force events are not force decomposition.

Decision: reject this rollout64 recipe, no automatic extension. Three-way
comparison63741 exits0, SHA2a6fbb44b23404aec8aa9dc3ff4bedd4d98e0e116f4413e6fe0e86c0998f1114.
CPU summarybafd22f26aabdbd5f80ef7afd42641b584afe804942139dee890f7457695418d;
network09afe3f03d5e51e217c9f6881bfb90449ab7756c49166c21a99617f49ff141fd.
Artifacts normal_rollout64_v1/OUTCOME.md. PICO saved-state rendering50341 completed
exit0: independent ffprobe decodes1122frames/22.44s/50fps/1600x688H264; previews
0/700/1121 inspected. Final2.260m root drift visible; no new dynamics or fabricated
return. Video normal_rollout64_v1/cpu100_v1/pico/render_view/source_vs_measured.fixed-world-v3.mp4,
SHA df372a30f6d4d1c0d7bb8205a4cec3a4b14835fa93895850420d8cb1ed4a669a.

Next baseline gap: recorded all29 PICO test uses low-latency teleop200ms; current
stronger native23 duration baseline uses normal teleop920ms. Earlier normal29
tests used G1 encoder. Need a matched all29 normal267-input teleop/920ms trial
before more adaptation, without relabelling those older controls or assuming
the normal29 policy will pass. This is not a proved source defect or a new result.
Goal ACTIVE; live full-body teleop NOT READY. No robot/DDS/modes, export, commit/push.

## 2026-09-09 21:15 UTC: execution parity checked; separate rollout64 experiment started

Read-only normal-core CPU/GPU audit40657 exited0. Reloaded checkpoint0/100
exactly reproduce saved128-row GPU means. All7614 CPU evaluation attempts have
bit-exact tokens on GPU batch1; raw maximum difference1.90735e-6. GPU batch128
changes two walking002 tokens, max raw0.0471342; other four cases' tokens exact.
PICO all2218attempts have exact tokens in both GPU modes, max raw3.81470e-6.
The existing2e-6/1e-5 diagnostics remain separate and unrelaxed. This does not
establish trajectory or hardware parity, nor explain metre-scale PICO drift.

Actual normal training used128env x16controls, unlike older low-latency32x64.
Normal16 has12688nonterminal rollout cuts; at16controls before a failure,1236
eligible uninterrupted episodes have no direct same-rollout failure credit.
Beyond0.32s depends on critic bootstrap. All667/819/546 source anchors were
sampled; missing source-phase coverage ruled out. See normal_execution_parity_v1/
OUTCOME.md and rollout_report.json; no claim of causal proof from diagnostics.

New separately named train_g1_true23_normal_rollout64.py keeps actor/rewards/
limits/920ms reference/seed/rates, changes physical rollout geometry to32x64,
and captures actual PPO bootstrap/returns via a hook without extra inference.
Only2x4x64 smoke or100x32x64 full run allowed. Focused tests89186:15passed.
Actual smoke11706:512transitions,2updates,16optimizersteps, exit0. Independent
reward audit77146 exits0; all PPO stored rewards exact. New return audit exits0:
all captured GAE-derived returns/normalized advantages bit-exact on fresh GPU
arithmetic. Initial actor AND critic states equal predecessor bit-for-bit;
initial simulation states are not asserted equal. Frozen core unchanged.

Full run46776 started under normal_rollout64_v1/pipeline_v2.py:100updates,
204800transitions,800optimizersteps, then five full original CPU lifecycles and
independent audits. Outcome pending; no motion success or promotion claimed.
Original pipeline.py failed before subprocess creation on a logging Path type;
empty first receipt/source preserved, corrected wrapper has separate v2 receipt.
No old source/limits/dataset changes, robot/DDS/mode commands, export or commit/push.
Goal ACTIVE; live full-body teleop remains NOT READY.

## 2026-09-09 20:45 UTC: normal-core100 fully tested; PICO37.34s regresses, checkpoint REJECTED

Fresh native23 original-core LoRA/Root9 training completed100updates/204800
transitions; real parameter updates and frozen-base identity independently
verified. Full CPU evaluation68338 exited0: walking002/003/008/dance complete
1417/1569/1114/1296controls, but all original world tracking screens FAIL.
Root p95 original→adapted:0020.583985→0.644377m;0032.483157→2.489729m;
0082.346464→2.739878m;dance0.906232→0.651980m. Dance28.06% improvement does
not offset the walking regressions or qualify its remaining hand/foot errors.

Held-out real PICO stops2217/6530totalcontrols,1867/5780sourcecontrols=37.34s,
versus untouched original core50.60s. SAME1867source-prefix root p95 worsens
1.791865→2.023562m and leg RMSE0.152214→0.166163rad. Do not compare unequal
durations against the baseline's longer-prefix5.24m score. Return/standing not
reached. Optical-derived PICO/OptiTrack reference, not live/sensor-only inference.

Fresh network audit80526 exited0: all7614attempted inputs/Root9/histories/tokens/
raw actions and7613accepted actions exact; final PICO guard rejection reproduced.
Actual saved-target physics all76130substeps reintegrates qpos/qvel exactly0,
no intermediate state writes. Actual hard range and applied effort excess0.
Separate zero-warmstart preview2e-10 check FAILS onwalking002 at4.86e-10 maximum;
threshold unchanged, failure retained, remaining4cases pass. First strict audit
5111source archived. Empty force events do not prove force decomposition.

Saved PICO failure localization: left ankle pitch−0.848155rad, velocity−3.793114
rad/s, hard lower−0.87267rad; all8guard candidates predict crossing. Actual pose
inside limits; final guard does not explain earlier drift or hardware damping.

Decision: no promotion, export, or automatic extension of this100update recipe.
Original untouched core remains better PICO baseline. Artifacts normal_core_adaptation_v1/;
OUTCOME.md records exact comparisons, preserved metadata failure, tests and
remaining failed preview check. Summaryf8dff0386d61057301cf70b7577e89411d4a52763190b44116a17a47fa7a6cf7;
networka23124a281f8f982ccdc49d3a7ad1f10776bb5ba08b7f70999752ef3e974813c.
PICO source-vs-measured saved-state rendering5090 running. Goal ACTIVE; live
full-body teleop NOT READY. No robot, DDS, mode, limits, source rewrite, commit/push.

Rendering5090 subsequently completed exit0. ffprobe independently decoded all
2218frames/44.36s/50fps/1600x688H264; previews0/700/2217 inspected. Fixed-world
view shows final2.761m root error. File normal_core_adaptation_v1/cpu100_v2/pico/
render_view/source_vs_measured.fixed-world-v3.mp4;
SHA712ae3606cf95f399a83bb3dd31610da86d91cbe1377536f4449ad78ec530023.
No new controller trial or fabricated ending. Full detailed disposition in
normal_core_adaptation_v1/OUTCOME.md. Current bounded recipe closed as an upgrade;
remaining goal requires a different evidence-backed adaptation step, not more
unchecked training or physical dance trials.

## 2026-09-10: distinct original-core native23 trainer executes actual SIM updates

Follow-through: fresh100update run94303 completed exit0 (204800transitions,
800optimizer steps;4m15s reported training loop). Checkpoint34963afbe664cf5a2e82b2d11686b4756c22ffa6dfbe469a2d614aae17fb7c24.
Independent saved arithmetic audit9085 exited0: every LoRA A/B, Root9 and
exploration changed; source encoder/decoder bit-exact. All PPO stored rewards
reconstruct exactly; maximum base-component reduction difference1.53e-5 within
predeclared5e-5 audit tolerance.1286terminated and81timeout transitions, including
1056world-root failures. These are training resets, not completed-motion scores.
Audit966e7686eccc3043428e71c01c34aaabbdabc908e6a791436cd053721a36540d.

Full CPU evaluation uses cpu100_v2. First attempt85670 exited1 after its first
simulation, during report construction (duplicate dict keyword 'kind'); no
usable controller result recovered. Exact faulty sources and started manifest
preserved under failed_report_metadata_v1 and cpu100_v1. Corrected metadata
merge, checked before physics, with regression test. No changed inference,
weights, constraints or grading. Updated focused suite32427:27passed, exit0.
Repeat68338 running; no resulting motion success claimed yet.

Implemented a separate seven-affine rank16 LoRA + Root9 actor/runner and explicit
47-sample920ms source path. Old low-latency executed sources remain unchanged.
Actual source-weight preflight1692 exited0:47saved walking002 observations match
untouched original-core outputs exactly, no global RNG consumption. Two disposable
gradient checks are connectivity evidence only; those test weights were discarded.

Actual GPU simulator smoke91228 exited0:4envs x8controls x2updates=64transitions,
16optimizer steps. Initial mean on actual simulator observations matches original
base bit-for-bit. Root9, decoder adapters, exploration and critic all changed;
frozen source encoder/decoder unchanged. Native scene and action/history checks
pass. Actual original920ms lower/VR observations are exact; maximum Root9 component
error1.49e-7, original task point error1.80e-7. No extra physics or state mutation
in those parity checks. Focused suite57386 exited0:23passed; scoped Ruff passes.

Artifacts C:/Users/camer/sonic23_sim_artifacts/internet_pico_20260909_v1/normal_core_adaptation_v1/.
Smoke checkpoint9dcb056cf991abd6871a743bcd2031f2eca7065b7854771306dfc8991e184d9b;
outcome8107ac1094f3b43ac27ab06a63b5bed78f3c1a0608fe91141186c6b914faa566.
Fixed-observation mean changes0.0618472maximum; NOT a tracking improvement score.

Next explicitly bounded experiment: fresh100updates/128envs/16controls, same
verified002/003/dance corpus, no PICO or008 optimization, followed by complete
original-core matched CPU lifecycle and PICO evaluation. No automatic training
extension on an unqualified checkpoint. Goal ACTIVE; full-body live teleop and
normal physical handoff remain NOT READY. No robot/DDS/mode/limit changes,
hardware export, candidate promotion, commit or push in this work.

## 2026-09-09 19:30 UTC: original-core PICO reaches50.6s, independently verified; full tracking still FAILS

One new complete-reference CPU request95631 ended0 with a preserved controller
guard failure:2880/6530 controls,2530/5780 source controls,50.6s source and57.6s
total. Earlier quality100 native23 reached13.9s source. Same original optical
PICO sample, source speed, native23 model/limits and scored phase; no training.

Exact common695-source-control prefix: original-core root p95=0.534082m versus
quality1001.322814m (59.63% lower), leg RMSE0.118125 versus0.154635rad (23.61%
lower). Left/right-hand centered p95=0.112307/0.104167m versus0.110468/0.132679m;
left-hand position slightly worsens. These are prefix diagnostics, not full
motion scores. The new run's longer2530-control source prefix root p95 grows
to5.239931m. Return/standing phases never reached; all absolute world screens
FAIL. Longer run does not qualify full-body tracking or justify physical use.

All28800 saved physics steps independently reintegrate with qpos/qvel differences
exactly0 and no intermediate state writes. Fresh checkpoint reload in audit13513
ended0: all2881 attempted encoders, measured histories, tokens and raw outputs
exact;2880 accepted targets and final guard rejection reproduce. Actual hard
range/applied/engine effort excess0; target jump1.232614rad/20ms remains
unqualified. Empty physics-audit events mean no constraint-force decomposition.

Artifacts C:/Users/camer/sonic23_sim_artifacts/internet_pico_20260909_v1/normal_core_pico_v1/.
Report112f4f5163ffe165c03910682c7e71226f805322a6535fea12d507d748e3b1d1;
network audit9238a660dfc00e429c5a235fa5fdda2b1d27732897700042a5ecdf3c8d0164f0.
Full saved-state fixed-world replay rendering78587 still pending at this entry;
display-only initial Path/str setup failure preserved, no controller rerun.

Decision: retain original core as a stronger measured simulation starting point,
not a deployment solution. This changes whole core plus required920ms reference
horizon and removes the earlier trained Root9; not a one-variable ablation.
Geometry and actual tracking both remain unresolved. No robot, mode, DDS,
training, reference rewrite, limit change, export, promotion, commit or push.
Goal ACTIVE; full-body live teleop and normal physical handoff NOT READY.

Render78587 subsequently completed exit0: normal_core_pico_v1/render_view/
source_vs_measured.fixed-world-v3.mp4, all2881 states/57.62s/50fps independently
decoded with ffprobe. Preview0/700/2880 inspected; fixed common world camera
shows final5.484m root displacement. Left prescribed29/right measured23 SIM;
no physical footage or fabricated return phase. Video SHA
eb7fb6c1b605f5a73ec7ceda4dbd32cf8c49e68a6e948bb40cdf61f7398b3397.

## 2026-09-09 19:17 UTC: missing-axis geometry measured; original-core PICO check started

Completed read-only diagnostic46043, exit0, all five full source references.
New g1_true23_missing_axis_geometry.py and eight focused tests separate exact
retained-pose native23 geometry from already recorded actual tracking. No
optimizer, new physics or robot commands in this measurement. Scoped Ruff passes.
Combined geometry/core/horizon regression suite19481:19passed, exit0.

Original dance546 frames: row-deleted retained-pose native hand gaps p95
0.297499/0.291540m versus source29. Original-core actual hands deviate
0.257217/0.275304m from that native pose, with total original-task errors
0.355668/0.354373m. Do not add these percentiles or claim a causal percentage.
Exact vector/MSE closure includes the signed cross term. Fixed retained poses
are not a reachability bound; other retained-joint motion may compensate.

Complete PICO5780-frame geometry gaps0.201658/0.209376m. Its actually completed
695-source-control prefix has native geometry gaps0.129740/0.127781m and
controller-to-retained-pose errors0.127557/0.135967m. Actual hand errors can be
smaller because components cancel. This pelvis-centered measurement removes
translation; it does not excuse the audited1.322814m root p95 drift.

Initial diagnostic9407 failed a false zeroed29/native23 FK equality before any
case output. Original source bytes archived. Corrected analysis measures the
small torso-frame/head-proxy offset separately; neutral head mesh offsets
cancel, so no physical1cm head defect is inferred. Both models, landmarks and
historical tracking scores unchanged. Summary SHA
b18578ba8dc957b1d3aeb384c620b72476cd22d644cf09f6493f139e6dd5c2cf.
Artifacts C:/Users/camer/sonic23_sim_artifacts/internet_pico_20260909_v1/missing_axis_geometry_v1/.

Decision: neither deleting six rows nor perfect tracking of row-deleted poses
establishes original task parity. Before constructing a separate normal-core
trainer, one new CPU test in normal_core_pico_v1 requests the same complete
PICO reference with the unchanged original SONIC core and its required920ms
received-source delay. Previous core test did not include PICO. No new
retargeting, speed change, Root9, limit adjustment or PICO training labels.
Started session95631; outcome pending at this entry. No readiness inferred.
Goal ACTIVE. Full-body live teleop NOT READY; physical robot untouched.

## 2026-09-09 18:51 UTC: whole-core comparison AUDITED; original core4/4, low-latency3/4; tracking still FAILS

Different hypothesis, no additional training: compare untouched original SONIC
encoder+FSQ+decoder against untouched low-latency core, each with its own correct
reference spacing, on unchanged complete002/003/008/dance lifecycles. No trained
root-feedback head in either. Verified whole-state identity against each approved
native23 initialization; all parameters frozen. Both configs select original
g1_model_12_dex; original C++ oracle verifies retained defaults, scales and row
mapping. Native23 physical plant, gains, effort, raw bounds and preview unchanged.

Original step5 needs47 received50Hz samples: positions0,5,...45, adjacent20ms
velocities through46;920ms delay. Not51 samples or100ms velocity differences.
Low-latency remains11 samples/200ms. Same delayed anchor9+t and scoredq2=11+t.
Normal simulated transmitter adds46 explicit synthetic terminal-standing frames,
low-latency10. No source trimming/retiming, EOF padding, measured-state rewrite,
root-feedback addition or live timing qualification. These are not raw PICO feeds.

Campaign96362 terminal0; eight actual CPU attempts completed. Root p95/leg RMSE:

| Full requested lifecycle | Original core | Low-latency core |
| --- | --- | --- |
| walking002,1417controls |1417;0.583985m/0.181439rad |528;guard rejection;no full-source score |
| turning003,1569controls |1569;2.483157m/0.206868rad |1569;2.073587m/0.191097rad |
| walking008,1114controls |1114;2.346464m/0.179519rad |1114;3.050258m/0.204005rad |
| dance,1296controls |1296;0.906232m/0.197510rad |1296;1.709359m/0.225615rad |

All eight unchanged source world screens FAIL. Completion does not mean faithful
dance or returned-to-requested-standing. Original core final standing root errors
0.830525/2.589179/2.264317/0.835642m. Actual measured hard-range excess0 in saved
results; this alone is not safety qualification. Original uses preview once on003;
low-latency twice on008. No limit or acceptance threshold relaxed.

New normal_source_horizon.py, g1_true23_released_core_comparison.py and focused
tests:11passed, scoped Ruff passed. First setup87086 was interrupted before any
physics: source inspection caught float32-reference versus float64-referee strict
comparison. Byte-exact original sources and empty output folder preserved in
interrupted_setup_v1. Corrected check converts only comparison operand to the
existing float32 source representation; regression tests cover both profiles.
No physical-motion retry occurred. Final run driver SHA
d363de92defa3a2e25724ceadbef6498357b35a1028b4ad5bdade275d40e11c5;
trials SHA482fda7ca90944e4500a9572da89d8b5a95546fdee6fd54835464f8a26e2c183.

Artifacts C:/Users/camer/sonic23_sim_artifacts/internet_pico_20260909_v1/released_core_comparison_v1/
Independent audit14278 terminal0. All9,904 attempted network outputs (including
the rejected walking002 output), original received encoders, measured-state
histories and guard decisions reconstructed exactly. All99,030 completed CPU
physics steps independently reintegrated without intermediate state writes;
qpos/qvel differences exactly0. Saved applied/engine effort and hard-range excess
exactly0. No constraint-force decomposition claimed: helper event list was empty.
Comparison SHAacb62af37a8be2f9c6455eac5c2d1d3d635fd04de76c4a86d33569d35146eeac.
The comparison's simple_normal_core_substitution_rejected:false is ONLY the
completion-only screen (normal4/4); all_absolute_source_world_screens_passed:false
and candidate_promoted:false remain. It is not a deployment/adoption pass.
Normal target jump reaches1.232614rad/20ms (008); dance0.824432rad/20ms, unqualified.
Normal dance root error improves46.98% versus untouched low-latency; turning003
worsens19.75%. Pelvis-centered dance hand errors0.355668/0.354373m and hand world
orientation errors1.482746/1.525408rad remain; better root drift is not hand-pose
parity. Outcome.md contains the final decision and full-versus-prefix distinction.

This comparison is of whole policy families plus required latency, not
an encoder-only ablation or evidence that low-latency equals the transfer paper's
fine-grained manipulation checkpoint. Normal may be a better simulation starting
point for some tasks, not a solved controller. PICO result remains FAILED13.9s.
No robot/DDS/deployment/mode actions, training, export, promotion, commit or push.
Goal ACTIVE; full-body live teleop and physical handoff remain NOT READY.

## 2026-09-09 18:16:57 UTC: failure-weighted sampler100 CLOSED FAILED; actual motion comparison audited

Fresh100 updates/204800 transitions completed; update audit passed, not control
qualification.1,869 source failures independently reconstructed;1,844 reset
anchors changed from uniform. Initial actor+critic exactly Q0, frozen weights
unchanged, actual LoRA/root/std/critic updates. This new reset distribution did
not solve tracking. No extension of this recipe, no physical trial or PICO rerun.

CPU002 stopped527/1417 controls (Q100 had completed). CPU003/008/dance complete,
but source root p95 remains2.401135/3.485536/1.733878m. Matched Q100 values
2.745039/2.965044/1.658840m: turning improves12.53%, walking008 worsens17.55%,
dance worsens4.52%. All completed absolute world screens fail. No truncated
prefix averaged into a full-motion score. CPU leg RMSE.175721/.197367/.223890rad.

GPU002 guard stops525/1417. At864, walking008 root-input difference2.026557922e-6
exceeds unchanged2e-6 consistency check and aborts remaining003/008/dance together.
Their completed counts864/1569,864/1114,864/1296 are validation-aborted requests,
not three robot falls. Rejected input vector not captured; exact discrepancy
computation not reconstructed. No threshold relaxed or live/physics retry.
GPU002 root failure began423, before guard525 and later global abort864;
003/008/dance first world failures464/418/515. Actual tracking was already poor.

CPU audit91581 terminal0:4,506 completed outputs+one rejected output, all original
received/history inputs,and45,060 actual physics steps independently replayed.
All token/raw/nominal-target and qpos/qvel differences exactly zero; guard failure
reproduced. GPU audit61757 terminal0 validates31,170 saved substeps/PD/limits/world
arithmetic; no independent GPU reintegration claimed. Captured actual hard-range
and applied/engine-effort excess zero. Maximum CPU target jump1.106922rad/20ms
still unqualified. Finalizer50240 terminal0 rehashes all bound evidence; decision
6a524bae5fdcf00ad5c2a99939b61b331b3e0e583900d8d0e0ca7aebe1ea8a39:
continuation_gate_passed:false, recipe_rejected_no_extension. CPU3/4 complete;
GPU0/4 complete, with guard failure versus validation abort kept separate.

Initial evaluation24302 failed metadata setup before physics; preserved nine
logs/commands/start records in failed_setup_v1 with byte hashes checked. New
metadata-only adapter and_v2 entrypoints fixed missing release_compatibility;
same actor object proven by actual-checkpoint preflight. Original trained reader,
policy and old failed source versions unchanged. Corrected campaign93417 ended1
on the recorded GPU consistency abort; saved evidence independently audited
without another run. Tests12+22+7 pass. No active process remains from this trial.

Artifact C:/Users/camer/sonic23_sim_artifacts/internet_pico_20260909_v1/failure_sampling_v1/OUTCOME.md
Checkpoint25360d13d547d221518131c75be153d620c086f99c2ff1b419fdca5420f2eb6f.
Update audit4befd73df9d86b5b9293409d592039371d079e02fb0151ad1a7ffd28a6053b88.
CPU audite316256a04ec9956d2288cfb984d04fc6505235ca05f217c2cded9ddd3b6e08a.
GPU auditdadf062cf965ecc2ae0c972b6c376e32be84da9461d597a7c2a488521684991a.
No hardware/DDS/mode/limits changes,exports,promotion,commit or push. Genuine
PICO test still FAILED13.9s source. Live robot-world-state estimation/timing and
physical handoff unproven; physical damping cause unresolved. Goal ACTIVE,
NOT READY. Do not relaunch uniform-Q, projection-loss, range-penalty, root-gain,
encoder-swap or this failure-sampling recipe as new progress. Need a materially
different evidence-backed policy strategy before more training; failed numerical
GPU identity does not justify discarding physical/simulation acceptance checks.

## 2026-09-09,17:27 UTC (03:27 Sydney Sep10): failure-weighted reset smoke verified; fresh100 started

New bounded hypothesis after the actual root/hand/foot failure capture: restore
failure-based source sampling, which our lifecycle command previously replaced
with uniform sampling. This is not another range penalty or range termination.
Same Q0 actor/critic, seed20260803, three training sources, reward, termination,
physics, action limits and optimizer.25% standing starts and uniform clip draw
unchanged. Other resets use per-source50-control failure bins, EMA.01, backward
weights[1,.5,.25],20% frame-uniform support. Same single RNG fraction; no extra
draws, cross-clip sampling, retiming or mid-episode state writes. No PICO/008
failure labels enter training. Separate checkpoint kind and validated sampler
state; old exporters remain rejected, no resume or deployment acceptance added.

Fresh2-update smoke completed4096 transitions. Independent NumPy reconstruction
verified every failure-bin update, EMA, probability and selected reset anchor;
39 true source failures,38 weighted source samples,all38 differ from uniform.
Initial actor AND critic exactly Q0; frozen weights unchanged, actual LoRA/root/
exploration/critic updates verified.12 sampler tests pass. Audit passed:true,
SHA ffb85e28bdaf95217758583a58cadf0b07f946875d9d0c488889da39daa9cb13.
Smoke sessions44362/45448 terminal0. Verifier retains one E501 style-only long
line rather than rewriting the hash-bound executed proof. No convergence or
physical motion improvement inferred from smoke.

After rehashing every smoke-audit input, fresh100-update run launched17:22UTC,
session10665; startup still ongoing at entry time. Not a resume of smoke or Q100.
Artifact C:/Users/camer/sonic23_sim_artifacts/internet_pico_20260909_v1/failure_sampling_v1.
Predeclared full CPU/GPU test:002,003,008,dance; all complete,zero range/effort
excess,mean source root p95 at least10% better than Q100,and no individual root
p95 or leg RMSE over5% worse. No extension if this fails. New strict reader
adapters reuse the unchanged complete-lifecycle measurement engine and corrected
rejected-history audit; evaluation starts at standing with no training resets.
Full-motion evaluation not run yet. No hardware/DDS/mode changes,exports,
promotion,commit or push. Genuine PICO test remains FAILED13.9s. Goal ACTIVE,
NOT READY; this trial is a new learning hypothesis, not a claimed solution.

17:47UTC update: fresh100 finished204800 transitions in15m18s training time;
session10665 terminal0. Independent update audit passed:1,869 source failures,
1,854 weighted source samples,1,844 anchors changed from uniform. Every captured
EMA/probability/reset independently reconstructed; initial actor+critic exactly
Q0,frozen10encoder+18decoder tensors unchanged,and all18LoRA factors/root/std/
critic actually updated. Candidate SHA25360d13d547d221518131c75be153d620c086f99c2ff1b419fdca5420f2eb6f;
lineage37456353b7ea871a6f683dea8ff705631125f0f5e9eb50a0c9a899b959dfc1ec.
22 additional evaluation-admission tests pass; new evaluation files Ruff-clean.
Evaluation session24302 active, first CPU motion starting; no motion improvement
claimed yet. Full evaluation reuses corrected rejected-history proof, not old
failed auditor retries. Read-only live-path inspection reconfirms older launcher
pins a different decoder; this root-feedback reader is SIM-only and needs a
verified live robot-world-state estimator plus timing/handoff checks. No claim
that connecting PICO alone makes this candidate deployable.

Evaluation setup correction: initial CPU002/003 failed before output directory
creation because strict new reader omitted release_compatibility from identity;
the geometry compatibility itself was already validated in semantics. Stopped
evaluation process group354 (only this local SIM campaign); session24302 terminal1.
CPU008 interrupted during setup; dance/GPU never started. No actual motion
rollout recorded. Preserved nine command/start/log files under failed_setup_v1,
verified identical SHA256 before/after recoverable move. Old evaluated source
versions remain untouched. Training checkpoint, source closure and successful
100-update proof remain unchanged; no retraining or resumption.

New g1_true23_failure_sampling_evaluation.py adds metadata from verified semantics
without changing actor; new CPU/GPU _v2 entrypoints and stopped-on-setup-error CPU
driver. Seven tests pass (session12811), actual100-checkpoint zero-physics-step
preflight passes (session79914); same actor object, original identity preserved,
both physical/source geometry hashes verified. New adapter/entrypoints Ruff-clean.
evaluate_after_metadata_fix.py requires this preflight and rehashes all successful
training-audit inputs before first actual full-motion tests. No weakening of any
model, source, safety or motion-outcome gate. This repaired evaluation plumbing,
not evidence of improved control or resolved physical damping.

## 2026-09-09,16:47 UTC (02:47 Sydney Sep10): actual stochastic training capture complete; range gap not dominant

Previous turn = progress. Completed its proposed passive actual-training probe:
fixed Q100, unchanged fresh32-env training factory,256 controls each,8192
transitions/81920 captured2ms physics steps,zero actor or critic updates. No new
termination or reward installed. Passive observer preserves actual pre-reset
states and reads existing termination results once. First setup failed before
rollout because MJLab names joints with robot/; exact namespace support added,
failed source/log/contract preserved byte-identically.15 tests and repository
Ruff checks pass. Separate opt-in range-termination recipe remains unqualified.

69 completed failed episodes:54 world-root and15 hand/foot height errors; no
simultaneous terms/timeouts. Only2 controls/11 substeps exceed hard joint bounds,
all left ankle roll in one walking episode; maximum.00557779379rad. Neither
crossing coincides with an existing done, but first precedes root failure by
120ms (controls243/244 versus249, env3). Root errors.227706/.237645m at crossings,
.313379m at failure. Other68 failures have no prior measured hard-range crossing.
No episode starts outside bounds. This is a fixed-policy fresh-reset sample,
not a retrospective occupancy count of original100-update training. It does
not quantify predictive-guard reserve rejections or justify another100 updates.

Independent measurement arithmetic passes: every captured hard-bound check,
old term OR, reset separation, root error and unchanged2ms-stale FK phase,
reference index, reward/PPO capture and pinned input hash. Strict observational
identity FAILS: observer/no-observer max8-control qpos delta8.99e-6. One bounded
no-observer repeat ALSO fails (max post-qpos1.886e-5); some derived initial
features/actions already differ despite identical initial qpos/qvel. Do not
claim observer independence or numeric engine parity. Final audit retains
passed:false with measurement_arithmetic_passed:true; no tolerance/gate relaxed.
Failed comparison scripts remain; verification-only v3 records failure while
auditing measurements. No further repeat or range-based training planned.

Artifact C:/Users/camer/sonic23_sim_artifacts/internet_pico_20260909_v1/training_range_diagnostic_v1/OUTCOME.md
Audit bbd23cc260b45595c66c9b440623dbe5eaffc111d057596911d5578a3469f381.
Sessions56947/44402/26052/29446 terminal0; no active process. No new policy,
hardware/DDS/mode/limit changes, export/promotion/commit/push. Genuine PICO source
test remains FAILED13.9s in. Goal ACTIVE, NOT READY. Next work targets actual
root/end-effector tracking, checks prior closed approaches, not another guard
or range-penalty loop. Missing range term is not the main observed failure mode.

## 2026-09-09,16:01 UTC (02:01 Sydney Sep10): finite-PD/range mismatch measured; no speculative training extension

Previous goal turn = progress (independently verified failed recipe). Fresh code
read confirms invalid_native_model_actuation checks raw/nonfinite conditions,
not measured joint bounds; joint_limit remains a soft reward penalty. Added
opt-in sonic_true23_measured_range_failure.py: records every2ms post-step state,
latches brief hard-range excursions, preserves existing termination terms and
does not alter state/torques/rewards/limits/step ordering. No existing training
recipe or checkpoint modified to silently adopt it. Eight tests and Ruff pass.
Actual MJLab rollout integration remains unverified, not inferred from fake-env
callback tests. A new explicitly versioned recipe would be required for training.

Replayed existing eight-candidate guard searches from saved CPU008/GPU002/GPU003
terminal states. Same native model/physics/search; no new inference or live trial.
All three rejections reproduce. Nominal ankle excursions.01258489/.00748195/
.01991549rad despite inside-range targets; finite-only PD invalid flag staysfalse.
Across24 candidates/240 predicted substeps,142 out-of-range substeps missed by
that finite-PD check. New latch flags23/24 candidates; remaining candidate fails
only stricter preview reserve. Measured latch is NOT predictive-guard replacement.

Crucial: both GPU runs already trigger world-root training failure before guard
stop: first control434 vs rejection525, and458 vs1129. Preceding recorded actual
training failure=true; root errors.402279/1.611735m. CPU terminal root1.489598m,
without recorded training-manager flag. This does NOT establish range omissions
as primary cause, actual training occupancy, or a reason for another100-update
run. Next: passive substep/state/failure capture in actual stochastic training
before old terminations, not blind reward/threshold/penalty changes.

Artifact C:/Users/camer/sonic23_sim_artifacts/internet_pico_20260909_v1/measured_range_v1/OUTCOME.md
Audit6f1c10fd06d6393bb584460a8b293b8e6726cafefbd9f6e39c39cc307a9d838e.
Sessions32296/95520 terminal0. No active experiment, new training, hardware/DDS/
mode/guard relaxation/export/promotion/commit/push. Goal ACTIVE, NOT READY.

## 2026-09-09,15:43 UTC (01:43 Sydney Sep10): reachable-mean LoRA experiment COMPLETE; rejected after independent verification

New coefficient.05 projection-loss recipe, same fresh initialization as failed
quality100, completed100 updates/204800 transitions. Independent actual-weight,
optimizer, reward and captured-mean loss audit passes.53 combined tests plus5
post-hoc probe and2 actual-terminal-input/tamper tests pass. Nine new repository
source/test files pass scoped Ruff. No extra training after the declared100.

CPU3/4 complete versus quality100's4/4; GPU2/4 versus4/4. CPU002 root p95 worsens
46.36%,003 root improves14.76% but legs worsen10.55%, dance root worsens5.74%.
CPU008 guard stop541/1114controls; GPU002525/1417 and0031129/1569. Dance completes
1296controls on both engines but fails original world tracking. All completed
world screens FAIL. CPU/GPU continuation gates FAIL; do not extend this recipe.

Independent CPU replay verifies4823 completed outputs, the extra rejected raw
output, and48230 physical substeps exactly. Failed guard reproduced at saved
terminal state. GPU saved-trace audit verifies40640 substeps/PD/metrics, not GPU
physics reintegration. Actual range/effort excess zero only on executed steps;
target jumps up to1.232614rad CPU/1.427539rad GPU per20ms remain unqualified.

Audit-reader v1 KeyError preserved with original source/log/exit. V2 reconstructs
the real final pre-codec history from ten measured states/previous targets and
checks received encoder/root/timestamps plus all542 attempted raw outputs on008;
does not invent a missing terminal decoder token. Verification repair reruns no
training or motion trial. All processes terminal:58120=0,83027=1(preserved auditor
bug),78322=0. No active experiment process.

Post-hoc same-quality-state probe on all2396 source inputs separates policy
changes from changed rollout states: overreach loss002 .093228->.086344,003
.072454->.067380,008 .221778->.226498,dance .053327->.053637rad^2. Two improve,
two regress. This does not establish projection as the sole cause or a general
fix; it cannot alter the failed predeclared decision.

Actual100 audit e38d643d11020f9d44ac14b87065be65720123787f4868b13cac6b549a171402.
Checkpoint684cc4b0d0257cb820d6427afe8de781ff2c3bca2553f1848c1315017a06df27.
CPU audit1ad0c15f00b4ed22d7c859851a039f963c3ea8738f985ebea4d78aa318957cd6;
GPU audit b9a7da833d4645afd15148abca227be2de18a595cbd7019fccab476f9969b026.
Decision be87dbe32fb269fc95d066ddb338abbbf27b9c5b3bf252442f3847e4e684237d.
Artifacts: C:/Users/camer/sonic23_sim_artifacts/internet_pico_20260909_v1/world_projection_v1/OUTCOME.md
and comparison.json. Prior public PICO-FreeDancing test remains failed at13.9s
of source motion; no PICO training or new candidate promotion. Goal ACTIVE,
deployment NOT READY. No hardware/DDS/mode/guard-relaxation/export/commit/push.

## 2026-09-09,15:14 UTC (01:14 Sydney Sep10): reachable-mean LoRA100 trained; independent motion evaluation RUNNING

Real PICO source intake/test complete below, control still failing. Read current
23dofsonic task; its older readiness claims do not override later matched V14
failures. No repeat of old policy, encoder, or gain trials as new progress.

New read-only current-world-quality100 exploration audit: some Gaussian mean
needs target projection in60.72/60.44/81.04/67.22% of002/003/008/dance source
controls. Some joint has<1% probability of sampling inside the envelope in
39.58/36.39/65.38/41.21%. Computed from actual std and independently reconstructed
physical target bounds; projection agreement<6e-8rad. Evaluation occupancy only,
not training occupancy or sole-cause proof. AuditSHA
0c7e2c65bf8ca9a323e65c3147993d9206e6962a5fe12d87d32aef60d70f0041.

Current frozen-base LoRA family excludes auxiliary projection loss. Added new
explicit recipe reusing existing ProjectedTargetPPO physical mean-overreach
loss coefficient.05 from older full-decoder work. Not a novel loss or resume of
failed world-quality recipe. Same fresh actor/critic, Gaussian distribution,
rewards, optimizer groups/rates, three training references, seed, physical
model, limits, gains, guards, source speed and timing. Distinct strict snapshot
header/reader; no legacy snapshot relabel or live/export acceptance.

29 focused tests pass, including zero-coefficient exact upstream PPO parity
and inward-gradient behavior. Smoke completed2 updates/64 transitions, process
terminal0. Independent actual checkpoint/mean/loss/reward audit terminal0 and
passed: initial actor AND critic match quality predecessor exactly; projection
loss max reconstruction error1.41064e-8, projected-fraction error5.264e-10.
Actual smoke training projected-mean occupancy54.6875%, under1% probability
occupancy39.0625%. Two-update loss.1244 -> .2489 is NOT improvement evidence.

Single predeclared fresh100-update32x64 regression completed204800 transitions,
process58120 terminal0; training loop14m49s.53 combined training/reader/strict-
gate tests pass; new evaluation repository files and artifact wrappers Ruff pass.
CPU/GPU baseline schema/hash probe passes and correctly rejects unchanged
baseline as an improvement. Independent actual100-update checkpoint/loss audit
PASSED, terminal0: same initial actor/critic, intended trained/frozen partitions,
6400 rollout calls/800 minibatches/409600 evaluated minibatch means; auxiliary
loss max reconstruction error7.9865e-9. Training mean-projected occupancy48.93%,
under1% unprojected probability occupancy33.26%; first/last mean loss.09930/.08459
does not prove control improvement on different training states. Four complete
CPU/GPU lifecycles versus quality100 in evaluation session83027. CPU recordings
finished3/4 full lifecycles versus quality100's4/4. New root p95 002=.854190m
(old.583619),003=2.339777m(old2.745039), dance=1.754074m(old1.658840);
leg RMSE .170714/.202528/.225434rad.008 FAILED541/1114controls at unchanged
joint-range guard; all failed physics and extra attempted action preserved,
legacy recorder's later timing-shape rejection does not erase that real failure.
Dance1296/1296 controls includes return, but final root error1.64648m remains
unqualified. GPU finished2/4:002 guard stop525/1417,003 stop1129/1569,008 full1114,
dance full1296. Continuation gate FAIL on both backends; no extension. First CPU
audit passed002/003, then failed in added all-attempt reader: rejected attempt
has raw/encoder/root/native pre-codec history but no saved decoder994. Preserved
audit_cpu.py and failed log/exit; new audit_cpu_v2.py reconstructs terminal history
from last ten actual measured states and previous targets, verifies last received
source window, and checks all542 actual raw outputs without pretending the last
token was recorded. Two actual-data/tamper tests pass. Seven additional probe/
repair tests pass beyond53 combined tests. No motion reruns. Verification-only
continuation started via finish_evaluation_v2.py; formal decision pending. Added explicitly
post-hoc same-baseline-state mean-projection probe to distinguish penalty effect
from differing rollout states; it cannot change the predeclared decision. Continue
only if all complete, mean full-source root p95 improves>=10%, no individual
root p95/leg RMSE worsens>5%, actual range/effort excess zero; CPU/GPU judged
separately. Existing absolute/contact/rate/live readiness gates remain.
No PICO-FreeDancing training, hardware, DDS, mode commands, guard relaxation,
export/promotion, commit or push. Goal ACTIVE, deployment NOT READY.

Artifacts: C:/Users/camer/sonic23_sim_artifacts/internet_pico_20260909_v1/world_projection_v1/
PredeclaredEXPERIMENT.md; smoke_v1/update_audit.json; regression_v1/training.log.

## 2026-09-09,14:21 UTC (00:21 Sydney Sep10): real PICO optical -> released SMPL336 encoder tested; no native23 promotion

Distinct from already rejected fullpose640 switch: tested source human SMPL
encoder directly on SAME complete PICO-FreeDancing optical recording. Built
strict neutral SMPL-X adapter using actual upstream FK and points0..21,39,54,
not PICO24 hand-joint substitutes. Source already Z-up; no double-axis conversion.
Original nonzero neutral pelvis rest offset preserved. Four20ms frames, per-frame
72 local-point +6 relative-root +6 virtual-wrist channels, flattened336. No
absolute root feedback added, no sparse-sensor reconstruction claim.

Predeclared source29 gate; same6530-control lifecycle, physical model/gains,
low-latency checkpoint decoder,200ms received reference and existing raw guard.
Generated standing/ramp/return uses original teleop267; human source windows use
SMPL336, first control352. Original baseline first352 controls match bit-exactly.
No code/physics/guard edits to existing run_case.

FAILED1739/6530 controls:1389/5780 source controls,27.78s dancing. Rejected finite
right-wrist-yaw raw10.0411758423 against10; saved extra attempted output. No
physical fall/damping conclusion. Last pelvis height.769172m/root error.454164m.
Same695-source prefix root p95 improves.235754 -> .208399m, but leg RMSE worsens
.141361 -> .232717rad (+64.63%). Entire executed source prefix foot/hand/head
world screens all fail. Maximum target step1.219458rad/20ms remains unqualified.
Zero integrated joint-range/applied-effort excess. Return/standing not reached.

Independent5780-frame FK max2.988838e-6m; actual-measured relative orientation
max5.960464e-8. All1740 attempted outputs/tokens and17390 physical substeps
reproduce exactly.28 related tests pass; new repository utility/tests Ruff pass.
Artifact runner retains11 C408 style-only findings, not claimed fully lint-clean.
Executed source and bound physics unchanged. Gate failed; no native23 follow-on,
training extension, export/promotion, hardware access, commit or push. NOT READY.

Artifacts: C:/Users/camer/sonic23_sim_artifacts/internet_pico_20260909_v1/pico_freedancing_v1/smpl_encoder_v1/
OUTCOME.md contains full comparison; reportSHA573ff1a2df3b2546b443abb5a9cd98ea053dc6f8b4b1e4006ae9959b3162b8da;
auditSHA69d06be73d28a787ec406b2a761967041e8896ea9504743ae52ffeb00a8a9663.
Do not repeat this encoder substitution or previous fullpose640/gain sweeps
as new progress. Public motion intake now works; control/fidelity remains unsolved.

## 2026-09-09,13:57 UTC (23:57 Sydney): publisher PICO-FreeDancing motion tested; native23 and original29 diagnostics FAILED

Latest request: find real PICO/internet motion and test here. Found direct
PICO/ByteDance HMD-Poser publisher-linked FreeDancing sensor/optical pair,
female1/20231025_032712, selected first lexicographic complete pair before data
inspection. Only two members fetched with bounded public ZIP ranges;6936 paired
frames,115.5833s at publisher60Hz. Safe weights-only load; all finite. Optical
local rotations/world joints agree within2.6008e-7m. Sensor arrays saved separately.
This is not raw XR24 SDK telemetry or sensor-only full-body PICO reconstruction.

Full optical reference retargeted with pinned GMR bb1bbe40774794fceb2a7c579a3464a28e68c844
and unchanged published G1 config.5780 samples at50Hz, speed1.0, one constant
initial XY/yaw registration, no vertical/per-frame reanchor or clip selection.
All29 solved coordinates retained for virtual intent; correct native23 selection
for physical reference. Full standing/source/return request6530 controls.
First setup attempt failed before any IK: qpsolvers4.12 expected daqp>=0.8.2,
not0.7.2. Installed compatible DAQP in isolated artifact dependency path; unchanged
prep completed in optical_reference_v2. MuJoCo3.5 and native physical limits unchanged.

Native23 world-quality100 checkpoint + existing range preview failed1045/6530
controls:695/5780 source controls,13.9s dancing,20.9s total. Guard rejects right
ankle pitch; saved current−0.839544rad and−4.56961rad/s, nominal next20ms minimum
−0.927443 vs hard−0.87267. Eight bounded candidates all reject; not proof of
global recovery infeasibility. Final root error1.944m: tracking diverged before
stop. Zero integrated range/effort excess, target jump1.232614rad remains unqualified.
Independent1045 network outputs and10450 physical substeps reproduce exactly;
received input error0 and qpos/qvel errors0. Failed extra attempted actor row kept.

Original29 source-side diagnostic with SAME saved all29 lifecycle failed1704/6530
controls:1354source controls,27.08s dance,34.08s total. Existing raw-action guard
rejects finite left-wrist-yaw magnitude10.005159 against10. Fresh full replay
reproduces1704 controls/17040 physical substeps exactly and captures1705 attempted
actor outputs including rejection. Zero integrated range/effort excess. Does NOT
prove a physical29 fall or infeasibility of this human source; existing diagnostic
raw bound, different model/gains/actor/guards limit comparison conclusions.

Same695-source-control prefix (q2): root p95 native1.322814m/original29 0.235754m;
leg RMSE0.154635/0.141361rad. Both incomplete; no full-motion baseline or23-DOF
impossibility claim. Neither reaches standing return. No new training, gate
relaxation, promotion, live export, physical robot action or physical damping fix.

16 focused intake/FK/resampling tests pass. Standard Ruff22 findings all I001/E501
style; remaining rules pass. New executed source kept unchanged for hash-bound
evidence; no commit/push. Native video1046 frames/20.92s, both fixed-world panels,
all saved states rendered; previews0/700/1045 visually inspected. Left prescribed
GMR29 optical reference, right measured native23 SIM, not real robot footage.
All runs terminal0 (prep3694, native41447, render29120, guard54621, original10639,
original audit55780). No active experiment/training process. Goal ACTIVE, deployment
NOT READY. This completed source/test subtask does not complete full-body goal.

Evidence C:/Users/camer/sonic23_sim_artifacts/internet_pico_20260909_v1/pico_freedancing_v1/OUTCOME.md.
Native audit b5ff772ca769ac93f33af8a1fcb7283a6f58d8ec1f2c4d7b1693e8d4cb4ede63;
original29 audit3f0b842ac64c70c76df505475be2ee34ef32d172d61b3b78de24e6d5053d9d08;
video6df0845772bb2660ebfe4b6799d8e32e4fb03d3b77be4805280d4fe08572ea89.

## 2026-09-09,13:04 UTC (23:04 Sydney): joint task-quality experiment COMPLETE; rejected after full comparison

Final outcome: native23 CPU4/4 full lifecycles versus predecessor2/4; GPU4/4.
All CPU/GPU world tracking screens FAIL. GPU root p95 new0.777235/2.270125/
3.446593/1.747707m versus old0.769986/2.214977/3.174259/1.835328m. Mean root
worsens3.09%; walk008 worsens8.58%. Predeclared continuation gate FAIL: no
extension of this recipe. CPU completion gain and three GPU leg improvements
do not establish full-body tracking or live readiness. Independent CPU audit
reproduces5396 outputs and53960 physics substeps exactly; GPU saved-trace audit
checks53960 substeps, not independent GPU reintegration. All actual range/effort
excess zero; CPU target jumps up to1.268rad/20ms remain unqualified.47 quality
tests plus2 separate MOSAIC FK tests pass; scoped Ruff passes. Evaluation20375,
regression99171 and all other experiment processes terminal0. No active run.
Decision SHA5c452607232429ece559d7eb65b467ecbf06aa2c60f434cdee3a0d9813138d3e.
CPU audit bc329ad913f577c3939e0128d775c6f9121ca36b5d74bc5444a7ea4ff4c48804;
GPU audit2e99a01b749636964a6b6fa73ba0832a3f8e26c31386de01db89ef550a28065a.
See C:/Users/camer/sonic23_sim_artifacts/internet_pico_20260909_v1/world_quality_bonus_v1/OUTCOME.md.
Physical robot untouched; no live export/promotion, commit/push, damping fix
or new video. Goal ACTIVE; full-body teleop NOT READY. Earlier milestones below
describe the completed run, not still-running work.

Read-only audit of all204800 world100 transitions matches held q1 references,
clip and lifecycle phase. First10 versus last10 updates: root mean0.12855 to
0.14042m; foot raw cost10.73046 to12.71779; hand/head raw cost3.35043 to3.87110.
Actual per-transition reward1.53504 to1.50245 despite increased logged episode
reward.47.77% of late foot-cost samples have bounded-cost slope below1%.
This is reward sensitivity, not PPO gradients through physics or a causal proof;
changing policy/reset distribution confounds the comparison. Source joint/frame/
timing and separate actor/critic gradient clipping reconfirmed, no fix there.

Predeclared distinct SIM experiment adds only2/(1+combined world-task error)
per non-done control; zero bonus on termination/timeout. All21 previous base
terms, shaping, physical limits, failure predicates, observations, initialization,
optimizer and three training references unchanged. New strict snapshot header,
reader and actual PPO reward capture.28 core/checkpoint/independent arithmetic
tests pass37.54s, session85739 exit0; scoped Ruff passes. Independent auditor
reconstructs bonus from costs, original reward equations and actual stored reward.
Single2-update4x8 GPU smoke20897 completed0; auditor58417 completed0. All64
transitions reconstruct; bonus error maximum1.09e-7. Initial actor/critic exactly
match world-smoke0 and frozen source tensors stay unchanged. No terminal/timeout
occurred in this smoke, so those paths only have unit coverage so far. Audit SHA
8eaffc97023cbdf819389d1117d018bd7f1be3551db0d40c76d4cc9f921993a9.
11 further lifecycle gate tests pass22.33s,17840 exit0:39 tests total. Gated
100-update32x64 regression99171 completed at12:46UTC, exit0;204800 transitions,
15m05s reported training-loop time. Both initial and100 snapshots plus full reward
capture saved. Sequential independent update audit and four-motion CPU/GPU
evaluation started after terminal success; actual tracking outcome pending.
100-update independent audit passes at12:49UTC, including all204800 transitions,
1525 true failures(1316 world-root,209 other) and107 timeouts. All done bonuses
zero; actual PPO storage reconstructed; initial actor/critic match world0 and
frozen source weights unchanged. Audit SHA4832a1ae70128bf1a838b8096bd2e710ba0fd59000b36444f7b34fcdc83a2342.
Checkpoint100 SHA978ffb5eb3e52f7f14b82759c270631070d24a3131c5d07d75c37040ca012cfe.
Evaluation session20375 now in sequential CPU replays; GPU/independent physics
audits and continuation decision still pending.47 quality/evaluation tests and
two separate MOSAIC analytic FK tests pass. No tracking improvement claimed.
CPU campaign has since completed4/4 full lifecycles(1417/1569/1114/1296);
prior world100 completed2/4. New full-source root p95:
0.583619/2.745039/2.965044/1.658840m; leg RMSE
0.169133/0.183198/0.210299/0.221402rad. Old CPU002/003 were incomplete, so their
partial errors are not comparable full-motion baselines.008 root improves
3.21124 to2.96504m; dance1.77679 to1.65884m. Still not qualified tracking.
Planned standing return remains fixed terminal XY/heading, not a jump to origin;
no return-target fix was warranted. GPU full campaign now running in session20375.
Supplemental
EVALUATION_PROTOCOL.md fixes relative comparisons to completed GPU baselines;
earlier CPU002/003 prefixes cannot serve as full-source comparisons. No recipe
extension on failure. Evidence:
C:/Users/camer/sonic23_sim_artifacts/internet_pico_20260909_v1/world_quality_bonus_v1.
Learning audit SHA38ed659972c0303f75c5e55e6278d1a8b3ba6c2e083b6ad3836abc64ec4cf146.
Public TWIST2 references remain not verified rawPICO. Robot untouched; no DDS,
arming, handoff, export, live readiness, commit/push or physical damping fix.
Goal ACTIVE; actual full-body tracking and live teleop remain unqualified.

### Parallel read-only public-PICO source intake, same continuation

Found publisher-labelled PICO VR processed G1 references in BAAI-Humanoid/
MOSAIC_Dataset, revision55aa4c537fe1b9d0f5cdeb915f87c01ba46cfded, license
CDLA-Permissive-2.0. Preselected first file before reading arrays:250 frames,
50Hz,29 joints,30 body poses,449794 bytes. SHA521caae77299a9fe9aedaf14285dbea5ab525fdd909a031328cedb5618a2b993.
Both consistency gates FAIL: provided joints vs publisher-URDF body poses differ
up to0.06133m/0.68310rad; source velocities differ from position derivatives up
to4.34148rad/s. Body-rotation-derived angles reproduce rotations8.61e-7rad but
positions still differ0.23198m. No coherent all-channel source established;
preprocessing cause unproven. Two analytic FK tests pass. Download and failed
reports retained, no silent repair or policy trial. Not raw headset packets.
See C:/Users/camer/sonic23_sim_artifacts/internet_pico_20260909_v1/mosaic_pico_v1/OUTCOME.md.
Running quality regression, its training inputs and four-motion gate unchanged.

## 2026-09-09,11:39 UTC (21:39 Sydney): same-checkpoint full-pose encoder diagnostic COMPLETE; rejected

Distinct no-training source-side test after raw-history counterfactual failed.
Confirmed actual low-latency checkpoint contains G1 encoder640 and unchanged
shared original29 decoder. New SIM-only reader/input bridge reproduces exact
upstream flattened-q-then-dq reshape, retains all29 virtual source joints and
uses only11 already-received50Hz samples with200ms anchor delay.22 initial
tests execute original upstream observation/encoder code and test input/runtime
rejection/attempt retention. No live profile or physical motor state changed.

Predeclared EXPERIMENT.md fixes original29 baseline, exact saved002/003/008
references/speed/timing, same weights/decoder/history, model/gains/effort and old
run_case implementation. Only encoder267→640 changes. Three sequential fresh
lifecycles with one immutable CPU teacher finish1417/1569/1114 controls.
Root p95 old→new:0020.581253→0.590400m;0031.767552→1.236362m;
0082.470028→2.476050m. Leg RMSE ratios1.052206/1.090843/0.985152.
All3 world tracking screens FAIL. Mean root p95 improves10.71%, but two leg
regressions exceed predeclared5% limit: continuation gate FAIL, no native23
follow-on or further encoder variant. Source29 physics is NOT native23 physics.
These public TWIST2 references are not verified raw PICO headset captures.

Independent auditor40905 exited0. All4100 singleton outputs/tokens and41000
physical substeps reproduce bit exactly, including qpos/qvel/actuator force/
contact counts. Every received640input/history and original source/initial state
reconstructs. Full-source world metrics independently verify.003 actual hard
joint-range excess0.00766739rad on87substeps;002/008 zero. Source engine effort
caps hold, but target jumps reach2.15027rad/20ms and are not hardware-qualified.
4 continuation-gate tests pass; all26 combined tests pass in13.94s, session85657
ended0. Scoped Ruff passes. Recorder40786 and auditor40905 ended0. No active run.

Report SHA d566c28aa2a0a94f1e73a381052de49f5def63d0d1e3749e911e9cb9b65345f1.
Audit SHA95932e3f25716eb78e159f4bfb2960a42e294b1718f9cff4c472fe806b1a1567.
Evidence/decision: C:/Users/camer/sonic23_sim_artifacts/internet_pico_20260909_v1/
full_pose_encoder_v1/OUTCOME.md. No training/export/hardware/commit/push/new video.
Encoder-only swap does not solve tracking; no physical damping conclusion.
Existing native23 tests, standing return, realPICO/root estimation/timing and
hardware qualification remain failed or unproven. Goal ACTIVE; teleop NOT READY.

## 2026-09-09,11:09 UTC (21:09 Sydney): raw-request history counterfactual COMPLETE; rejected as deployment fix

Previous goal turn: progress, public full-motion tests completed/rejected and
evidence saved. Current turn investigates source semantics before more training.
Upstream CreatePolicyCommand stores raw network output in last_action. Current
native23 training/runtime uses projected target equivalents instead. New opt-in
SIM diagnostic changes ONLY retained previous-action inputs; actual bounded
targets, range preview, native physics/gains/effort, source/timing and weights
stay unchanged. This is a deliberate runtime-distribution counterfactual for
the trained actor, not a compatible checkpoint relabel or established bug fix.

V1 recorder missed previous_action metadata required by the old referee after
integration.002/003/008 reached that error; their transient traces were not saved.
No usable motion result claimed. Session62305/PID362 stopped with SIGINT before
dance integration. V1 source/error files preserved. V2 repairs only that label;
19 runtime/recorder tests and8 independent history-auditor counterexamples pass.

Fresh v2 session27982 completed all four attempts, with same world100 checkpoint
5ba74a79c9b25fbb75eff797b77ab2bf2a362d53afa94020ef3432e060faad22.
002 now finishes1417/1417,0031569/1569, dance1296/1296;008 now guard-stops553/1114.
Full-source root p95 for completed002/003/dance:0.68312/1.86989/1.71333m; leg RMSE
0.17052/0.20813/0.22390rad.008 partial1.92958m is NOT comparable to a full-motion
metric. All tracking screens fail. Final standing remains unqualified. Baseline
world100 completed2/4; raw-history variant completes3/4 but regresses008.
Trials SHA79edd9f19152bdabcfe8f0f679d2714074be21262bd0d27007483ac81bff7194.

Independent audit76938 ended0: all4835 executed singleton network/token outputs
and48350 physics substeps reproduce bit-exactly. Every raw/action/measured history
and received input reconstructs; source/timing/physical contracts match baseline.
All actual joint-range excesses zero.008 failed next proposal independently
reproduces; extra unapplied inference/history preserved separately. No force
decomposition events sampled in this audit; zero placeholder residuals are not
an additional force-balance measurement. Maximum target jump1.23261rad/20ms,
not hardware-qualified. Audit668e4e7a8970d1718a0f3846f77c6c118229d9fadd4dde9b5703e295d65321fb.

Matched source-prefix root p95 old→new:002174 controls0.76342→0.65382m;
003375 controls1.40363→1.61069m;008203 controls1.80441→1.92958m. Full546-control
dance1.77679→1.71333m. Short prefixes are not full-source improvements. Source
history differences reach4.20048 normalized units; exact differences start at
control1, so rounding effects are not separately isolated from clipping effects.
27 combined tests pass; scoped Ruff passes. Recording, audit and test processes
all terminal. Source-like raw history changes behavior but does not solve path,
contact or standing. Reject promotion and additional blind history/training loops.
This counterfactual also does not establish which history contract would work
best after consistent retraining. No default training/runtime profile changed.
No further training, policy promotion, live export, hardware command, commit or
push. Goal ACTIVE; full-body live teleop NOT READY. Evidence folder:
C:/Users/camer/sonic23_sim_artifacts/internet_pico_20260909_v1/source_raw_history_v1/.

## 2026-09-09,10:36 UTC (20:36 Sydney): world-tracking100 COMPLETE and REJECTED after CPU/GPU motion tests

Prior goal turn: implementation progress plus verified wait; interrupted run
preserved. Identical recovery100 completed204800 transitions. Independent audit
reconstructs every world error/condition and21 reward/PPO equations:1282 world
failures,248 other failures,107 timeouts. Initial actor AND critic match bounded0;
all28 released base tensors remain frozen. No source speed, limit or gate change.
Checkpoint5ba74a79c9b25fbb75eff797b77ab2bf2a362d53afa94020ef3432e060faad22;
update audit69b67c73f04466acb4190f222eebfd49d44b665ab6263af55feb793e2c1f9d5.

CPU:002 stops524/1417 and003725/1569 at unchanged range guard.008 and dance
finish1114/1296 but fail tracking, root p953.21124/1.77679m. Verification first
hit RAM allocation failure during concurrent GPU loading; preserved failure,
then identical sequential audit6378 succeeds. All36590 substeps reproduce
bit-exactly;3659 executed policy outputs exact; both terminal guard failures
reproduce. Unapplied extra proposal rows remain preserved, never called executed.
CPU audit66ce09ff9433b1b3381d2bb272dc312914617b078e8009aba00e9ee9872a4d2d.

GPU: all1417/1569/1114/1296 controls finish with no episode resets. All tracking
screens FAIL. Same four-world comparison against bounded100:

| Motion | Previous root p95 m | New root p95 m | Root change | Leg RMSE change |
| --- | ---: | ---: | ---: | ---: |
| Public002 |0.68417|0.76999|+12.54% worse|+7.66% worse|
| Public003 |2.82660|2.21498|-21.64% better|+1.70% worse|
| Public008 |3.20850|3.17426|-1.07% better|+6.20% worse|
| SONIC dance |1.95628|1.83533|-6.18% better|+2.11% worse|

All53960 saved GPU substeps/PD/limits and actual new failure measurements audit
cleanly. This checks measurements, not independent GPU reintegration. Source q1
and actual pre-last-substep pelvis match failure captures exactly. Final standing
joint errors0.288/0.356/0.321/0.343rad remain unqualified; target jumps reach1.185rad
per20ms on GPU and1.227rad on CPU. No physical damping/handoff fix established.
GPU report259fc21542c757e704008e775a3e2558da0e6632a6261055893c7cf9e123787d;
audit526639ebbf95b7f9ecf28925945b62eb8811991109898ae6bd81d2cab0fa252d.

Public003 video verified1570 decoded H264 frames,1600x688,50fps,31.40s container
(31.38s nominal integration plus initial frame). First/middle/final inspected;
fixed-world root errors0/1.516/2.394m visible. Original29 is prescribed reference,
native23 is measured SIM, not physical footage. Video:
C:/Users/camer/sonic23_sim_artifacts/internet_pico_20260909_v1/world_tracking_termination_v1/gpu_recovery_full_v1/walk003_video/source_vs_measured.fixed-world-v3.mp4.
SHAac515f45a15e1d5d6444e7e51cc24122726661679c60d0e7039cac93e5a5d716.

47 new tests plus28 existing lifecycle tests pass; scoped Ruff passes. Every
training/evaluation/audit/render process terminal. Mixed error improvements do
not justify promotion or another automatic100-update extension. Next investigate
source/control fidelity and reference feasibility before further optimization;
consult existing action-authority diagnostics to avoid repeating earlier work.
Original29 public buffered baseline also misses world screens, but model/gain
differences make it a source-side reference, NOT a controlled morphology ablation
or excuse to relax native23 gates. Real PICO packet/calibration/dropout behavior,
live root estimation, broad BONES-SEED generalization, timing, safe handoff and
fresh supervised hardware qualification remain unresolved. Goal ACTIVE.
No DDS, arming, robot command, live export, commit or push. Full-body teleop NOT READY.
Outcome: C:/Users/camer/sonic23_sim_artifacts/internet_pico_20260909_v1/world_tracking_termination_v1/OUTCOME.md.

## 2026-09-09,09:42 UTC (19:42 Sydney): interrupted world-tracking run preserved; identical recovery running

Prior goal turn: progress plus verified wait. New lifecycle recorder/auditor
passes23 tests; combined old/new lifecycle suite51 passes. No motion result yet.
At09:39 UTC, handle39486 was missing, WSL inventory contained no training process,
and only world_tracking_model_0.pt remained. TensorBoard last changed08:44:07 UTC.
No100 checkpoint/reward capture exists. Cause of process loss is not established.
All interrupted files retained under world_tracking_termination_v1/regression_v1.

Same capped100 recipe launched in regression_recovery_v1; ONLY run/curriculum
output paths changed. Not a resume, policy variation or extension of a completed
rejected candidate. Detached hidden Windows wsl PID19612 has durable stdout/stderr
logs and a launch receipt under world_tracking_termination_v1. Actual completion
must be established from process state and complete audited outputs, not receipt.
The new CPU/GPU comparison code remains unexecuted against a trained candidate.
No source/limit/safety/export/hardware change. Goal ACTIVE, full-body teleop NOT READY.

## 2026-09-09,08:28 UTC (18:28 Sydney): world-tracking training boundary implemented; controlled100 running

Prior goal turn: progress. Complete GPU public003 drifted2.8m while old training
height/tilt/end-effector-height failure predicates remained false. Added one
true training failure at world-root xyz error>0.30m (nonfinite also fails).
All old failures/timeouts,21 bounded reward components/weights/potential, source
speed/timing/data, physical limits/gains, architecture and optimizer recipe stay
unchanged. This is NOT a relaxed tracking acceptance gate or a hardware stop.

Fresh distinct SIM-only checkpoint header/reader; no existing candidate or live
launcher modified.17 environment/schema tests and7 independent audit tests pass.
Two-update4-env smoke23479 exits0; audit72440 passes all64 actual transitions,
including independently reconstructed world error and actual PPO flags/rewards.
Initial actor AND critic equal matched bounded0. Smoke max error0.17955m; new
failure does not fire in that short smoke, so terminal-path execution still pending.

One fresh100-update32x64 regression39486 launched; not an extension of rejected
bounded100. No tracking improvement claimed. All four complete original-speed
CPU/GPU motion tests remain mandatory; no success from shorter training episodes.
Experiment: C:/Users/camer/sonic23_sim_artifacts/internet_pico_20260909_v1/world_tracking_termination_v1/EXPERIMENT.md.
Full-body teleop NOT READY, goal ACTIVE. No robot command, DDS, arming, export,
physical damping fix, commit or push. Broader BONES-SEED generalization unresolved.

## 2026-09-09,07:55 UTC (17:55 Sydney): full training-engine motion test COMPLETE; all four tracking screens FAIL

Previous turn: progress from same-state/one-control boundary checks. Current
turn closes the full-loop gap; no retraining, hardware run or candidate promotion.
Unchanged bounded-progress100 now executes all four complete public002/003/008
and SONIC dance lifecycles in MJLab/Warp:1417/1569/1114/1296 controls,53960 actual
physics substeps.417 input bindings, same source speed/history/action codec,
native gains/effort/range guards; no intermediate state writes or automatic resets.
GPU source root p95 is0.684172/2.826601/3.208497/1.956279m. ALL full-source
landmark screens fail; final standing/contact unqualified. Duration completion
does not mean successful dance, teleop, physical return or damping fix.

Version1 startup stopped on its own invalid physical-replica equality assumption;
source and failure report retained. Version2 uses four independently recorded
physical worlds, not compressed32-world replicas. Physics batch4 vs training32
is explicit; batch-size invariance is NOT claimed. Actor inference remains32
through repeated actual observations, with every repeated raw/token output saved.
Corrected two-control smoke passes exact initial history/encoder/root/token
matching; raw max difference2.682209e-7. Full session92111 exits0.

Independent measurement audit4743 exits0: all53960 substep/control boundaries,
float32 native PD and effort saturation exact; source root/leg metrics reproduce.
This audit checks saved measurements, not independent GPU physics re-integration.
Zero actual range/effort excess; guards intervene twice each on002/008. Target
jumps still reach1.223084rad/20ms. Training failure predicates fire1/14 controls
on008/dance, recorded without episode resets; no ordinary-episode survival claim.
28 harness/audit tests and Ruff pass.

Full GPU/CPU loops diverge, but both fail world tracking. This rules out CPU
playback alone as the whole explanation; it does not isolate numerical causality.
No justification for hardware retries or another automatic100-update variation.
Next work must address learned tracking/reference feasibility using full-motion
screens. Public recordings remain retargeted TWIST2 data, not verified raw PICO.
Outcome: C:/Users/camer/sonic23_sim_artifacts/internet_pico_20260909_v1/training_engine_lifecycle_v1/OUTCOME.md.
Execution796358dee11223f808f8a301adbf4d269b507a326f187e4ab32746192d259cd4;
audit7455666893a0a499f38c6185ecad10fd143952ebc29f6f8bc885018934e0ba94.
Public003 fixed-world source29/measured23 video COMPLETE,07:58 UTC:1570 decoded
frames,50fps,31.40s container; first/middle/last images inspected. Final root
error2.818m is visible; original29 is prescribed, native23 is measured SIM.
Video: C:/Users/camer/sonic23_sim_artifacts/internet_pico_20260909_v1/training_engine_lifecycle_v1/full_v2/walk003_video/source_vs_measured.fixed-world-v3.mp4.
SHA0c653da02a8987ce076592c2400b18de59e42f88c35a8fd96ac982a8f410c592.
Codex panel opening queued. All experiment processes are terminal.
Goal ACTIVE, full-body deployment NOT READY. No DDS, physical command, mode
switch, policy update, export, commit or push.

## 2026-09-09,07:16 UTC (17:16 Sydney): measured training/replay boundary probes COMPLETE; no gross input or one-control engine mismatch

Previous goal turn classified as progress: rejected the bounded reward candidate
from completed training and measured CPU failures. Current turn closes a different
evidence gap; it does not retrain, promote or relabel that rejected policy.

New probe26334 reconstructs the actual training observation groups on3028 saved
measured controls:0021407,003715,dance906. It rebuilds ten chronological frames
from physical qpos/qvel and actual applied commands, not copied expected history.
Each case excludes its first10 synthetic-history controls; unapplied guard rows
are excluded. Shared training configuration, original29 source, native23 plant,
IEEE float32 and32 environments retained. Observation noise disabled explicitly
for this same-state diagnostic. State injection is diagnostic, NOT dynamic success.
All3028 FSQ64 tokens match saved singleton CPU tokens exactly. Maximum absolute
history difference1.907349e-6; encoder2.086163e-7; root9 7.748604e-7; raw23
3.784895e-6. Same-GPU input-construction and same-input backend effects are saved
separately.474 bound inputs; actor tensors unchanged;0 integration/learning.
Proof1c68e3e7062e29c046b474c71f0be77c535c7f85f317da63d4341966e3352d62.

Initial probe24705 failed in the diagnostic harness: CPU-reader actor constructed
for267 semantic values was invoked on the268-value training group. Versioned
wrapper restores the actual training route-index dispatch flag; no tensor,
recorded control, observation function or previous reader is modified. Failed
source and failure receipt remain. This is NOT a discovered deployment bug.

Physics probe14272 also completes. First recovers historical solver warmstarts
by bit-exact reexecution of all41720 completed CPU substeps. Then tests eight
evenly spaced states per002/003/008/dance, including endpoint controls:32 separate
20ms probes with ten2ms steps each. Inject initial state/warmstart once per probe;
no intermediate corrections. Actual saved targets held; native PD/effort caps
remain active. Compare GPU, attached CPU with native PD, and attached CPU receiving
the EXACT GPU motor controls. Static scene parameters match; all numeric arrays
and compiled training model retained,308 bound inputs. Largest GPU-vs-original
joint-position difference5.052218e-6rad, root-position vector difference
5.103316e-7m and joint-velocity difference8.459812e-4rad/s. CPU receiving identical
GPU controls still differs by up to1.041482e-3 in generalized velocity; no claim
of bit-exact engine parity. These are ONE-control probes, not full trajectories.
Proof965b21ada924bba04ca3a763d642b24d7fb9b8c7ac399ac60b91f2503927942e.

26 sampling/error-summary unit tests and Ruff pass. Outcome and exact scope:
C:/Users/camer/sonic23_sim_artifacts/internet_pico_20260909_v1/train_replay_boundary_v1/OUTCOME.md.
Decision: do not change joint mapping/export plumbing to fix a mismatch not
observed. Next substantive check is full CLOSED-LOOP tracking in the exact
training engine with matched startup/history and unchanged limits, compared
against existing CPU outcomes. Short-horizon agreement cannot exclude accumulated
contact/numerical divergence. If both fail, address learning/reference feasibility
with full-motion evidence, not another automatic100-update reward variation.
Full-body tracking, smooth return, live estimation/PICO/handoff, real-time control,
physical damping diagnosis, fresh supervised hardware evidence and coherent
verified commit/push remain unresolved. Goal ACTIVE; deployment NOT READY.
No physical command, arming, DDS, export, commit/push or policy update. Both
new runtime probes have exited; no related training/probe session remains active.

## 2026-09-09,06:39 UTC (16:39 Sydney): bounded-progress100 COMPLETE; four attempts audited; candidate REJECTED

Supersedes the ACTIVE note below.100 updates/204800 transitions finish in33953,
16m24s training loop, exit0.43 new reward/config/audit tests plus20 existing reader
checks pass. Update audit20356 reconstructs every actual reward/PPO record,
including123 failure and190 timeout transitions; all28 base tensors stay exact,
all18 adapter factors change. Initial48 actor tensors, critic, agent,32 reset anchors
and raw motion/model bytes match decoder-wide0. Implementation verified, NOT
full-body control solved. No resume or automatic extension.

New CPU4516 attempts:0021417/1417 and0081114/1114 complete but fail world tracking;
003 stops725/1569 at14.50s and dance916/1296 at18.32s when the unchanged range
guard finds no verified next target. Dance source ends896: stop is0.40s into
return, but source tracking already fails badly. Full546-control dance source
metrics are separately verified; no incomplete lifecycle is called successful.
Root q2 p95, first-affine100 /decoder-wide100 /bounded-progress100:
0020.656198 /0.861429 /0.604050m;0083.111962 /3.911588 /3.155324m;
dance source-only2.156039 /1.598540 /1.869780m.003 has no full-source new score.
New leg12 RMSE0020.172637,0080.210916,dance-source0.222709rad. Full completion
regresses4/4 to2/4. Final standing root maxima0020.522776m and0083.123598m;
standing/contact/live qualification remain false. Full original29 hand/head
world/relative/orientation metrics are retained in bounded_progress_v1/OUTCOME.md.

Audit73061 independently reproduces all41720 completed substeps bit-exactly,
with285 bound inputs,4172 singleton encoder/raw/nominal-target rows exact and
4132 actual post-seed histories verified. Both guard stops reproduce. Actual
range excess0;002 corrections527/528; other prefixes have no prior correction.
Largest target jump1.188512rad/20ms; no smoothness/full50Hz or recursive-safety
claim. All failed physics and726/917 attempted inference rows are preserved.

First batched audit8338 failed: walk003 frame659 has a0.0625 quantized-token
batch32 difference. Diagnostic45718 establishes exact singleton replay. The
versioned audit_cpu_singleton_v2.py uses original singleton shape on EVERY
frame, exact comparisons and unchanged policy/physics, not relaxed thresholds.
Original failure is retained. Other batch/backend/ONNX parity is unqualified.

Full public008 comparison video11018 completes1115 frames/22.30s,1600x688/50fps;
FFprobe and first/middle/final previews checked. Fixed-world source29 prescribed
poses versus measured23 SIM clearly show~3.1m drift. Not physical video or a
successful29-policy reference. Viewer open was queued, not assumed displayed.

Checkpoint100 SHA6efd755d281eb2d718e2a5c7ed1da75f4b9d3a5593c679fd855aae39d19c1cca.
Update/reward audit60779dc09fe207ebf5a5351372d511d4e941ec1e647469844ac03df2cb6b27d1.
CPU auditfa31b9a3f1f426de0c674ef6d864198d41f9133e41dd7028242fbee5ec96b3a6.
Dance-source proofa56b934449796a4b2c60806fe52de08080f04aa57cd7cf44cb522e03024e5d2b.
Videof0cafdc328e2bf2f67e1f37356af6f30f161d3c1389c6cd493c3a4954ff598e6.

Disposition REJECTED_NO_EXTENSION. Reward-scale flaw/transform does NOT establish
physical damping causality. No DDS, arming, mode changes, robot commands, live
export, commit or push. Public TWIST2 PICO-workflow references are not verified
raw headset captures. Full-body tracking/return, live estimation/PICO/handoff,
smooth control and fresh hardware evidence remain unresolved. Goal ACTIVE;
deployment NOT READY. No training, replay, audit or render session remains active.

## 2026-09-09,05:55 UTC (15:55 Sydney): reward-scale diagnostic and bounded-progress smoke audited; capped100 ACTIVE

Previous decoder-wide100 remains REJECTED; no extension or hardware action.
New saved-trajectory reward audit89031 completes10792 controls across all eight
first-affine/decoder-wide002/003/008/dance paths, with276 bound inputs and zero
new dynamics/inference/updates. Even granting every positive reward at maximum,
omitting other costs and granting a perfect infinite tail, selected original
world-tracking costs yield optimistic discounted suffixes as low as-15314.319,
versus fixed failure component-100. Training termination predicates were NOT
applied: this does not prove an achievable failure preference or physical damping
cause. Root anchor verified pelvis; held-q1 and original2ms-stale reward phase
preserved.22 return-math tests pass. Summary SHA256
6f96fc17f99107f52e56af3ec69ff6b2f8ca4db4f248fd0419c61a590ff3178c.
Full qualification caveats: reward_incentive_v1/OUTCOME.md.

Implemented separate bounded_progress_v1 research recipe, not a live fix.
Thirteen nonnegative raw costs become x/(1+x); alive5 becomes115.3, termination
stays-5000. Nonterminal base reward[0.1,2.406]/control; true-terminal base
[-102.206,-99.9]. Progress uses-log1p(selected world cost), sampled at synchronized
actor states before/after the unchanged step. True-terminal and installed RSL
self-bootstrap timeout cases are explicit; reset-state potentials never leak
into done-transition shaping. This intentionally changes the old optimization
objective and can still fail tracking. All original29 targets,23 physical axes,
actor architecture/initialization, gains/caps/ranges/terminations and CPU gates
remain unchanged. Distinct research checkpoint header/name and strict reader.

43 new reward/config/arithmetic-audit tests and20 existing reader checks pass;
Ruff passes. First unit run exposed float32 boolean-mask arithmetic in a float64
test oracle (9.5e-9 difference); corrected the oracle's dtype, not its threshold.
GPU smoke52088 finishes2 updates/64 transitions, exit0. Every actual raw cost,
base component, pre/post potential, returned reward, failure/timeout flag and
PPO stored value/reward is retained. Independent smoke audit19188 PASSES:
all64 reward/PPO records reconstruct (maximum component-sum difference4.37e-7),
all18 adapter factors update, all28 base tensors remain exact, old export rejects
both distinct snapshots. No terminal/timeout occurs in this short smoke; all
four flag cases are covered by separate actual-RSL arithmetic tests, not claimed
as observed events. Smoke audit SHA256
05515b952b1720b4b696b092e7f2b4252c746c22cc733e7794a58991e0a8e0e7.
Fresh capped100 regression33953 is ACTIVE,32x64/204800 planned transitions;
no resume/extension or controller improvement claimed. Four full CPU attempts
and independent audits are prepared, including both predecessor comparisons.
Initial comparison52668 PASSES: all48 actor tensors and all critic tensors are
bit-exact against decoder-wide0, all agent settings and32 reset anchors match,
complete lifecycle/reference/model bytes match. Only the reward objective is
intentionally different. Proof SHA256
c88cf0db068c360ba7dbd720eb6130a309345456be79f75c5348659a5cccf62e.

No DDS, physical commands, arming/mode changes, deployment export, commit or
push. Full-body live teleop remains NOT READY; goal ACTIVE. Public motion data
remains self-recorded TWIST2 PICO-workflow retargeted robot references, not
verified raw headset data. The bounded experiment and remaining full-motion
tests are predeclared in bounded_progress_v1/EXPERIMENT.md.

## 2026-09-09,04:55 UTC (14:55 Sydney): decoder-wide100 COMPLETE; all four full replays audited; candidate REJECTED

Supersedes the ACTIVE note below. The decoder-wide implementation and45 tests
are verified; fresh100 updates finish204800 transitions in session90621
(12m06s training loop). Saved update audit15819 PASSES: all18 adapter factors
change while all28 released base tensors remain exact. Checkpoint100 SHA256
2575912e3091ce79b5289290d00d13530cea1c25d342f33fd876598b868522ae.
No resume or automatic extension. This is implementation/evaluation progress,
not general full-body tracking improvement or live readiness.

Four NEW full CPU attempts89592 complete with the unchanged range guard:
public0021417,0031569,0081114 and SONIC dance1296 controls. Root q2 p95 changes
from first-affine guarded100 to decoder-wide100 as follows:
0020.656198→0.861429m (+31.28%);0031.956218→2.896564m (+48.07%);
0083.111962→3.911588m (+25.70%);dance2.156039→1.598540m (−25.86%).
Leg12 RMSE0.168893→0.178335,0.197609→0.190518,0.208114→0.233997,
0.215311→0.222411rad. Dance world error improves but relative limb errors worsen;
all three public paths regress. All four world-landmark screens fail. Standing/
contact remain unqualified, with final root maxima0.245960/2.923424/3.946295/
1.790752m. Full original29 hand/head world/relative/orientation comparisons are
retained in decoder_wide_lora_v1/OUTCOME.md; no missing source axis is dropped.

Actual hard-range excess is0 on all four, with0 guard interventions. Minimum
range margins0.059640/0.005587/0.014469/0.099753rad. Actual maximum velocity/rating
ratios0.381/0.573/0.448/0.492. However maximum per-control target jumps reach
1.034506/1.151032/1.168102/0.906626rad. Smooth targets and full50Hz timing remain
unqualified; preview timings alone do not establish end-to-end real-time control.

Independent audit14405 PASSES first execution:269 bound inputs,5396 received
encoder/root rows,5356 actual post-seed action histories and53960 reintegrated
substeps. Qpos/qvel reproduce bit-exactly without intermediate state writes.
Tokens are exact; batched raw error≤3.337861e-6 and target error≤1.668931e-6rad.
Original geometry/timing/task/standing metrics also reproduce. Audit SHA256
b1a4d0abbb9e0391f7c775a9ab06d6a7f48565b5131e23c0cd6e81945967e56a.
Update audit SHA256 aab4195abc3744759480aba44b539b846d2413d0c79b41bd8eceb161b40ffc99.
Trial receipt SHA25658e22de955feddc380330d6f4fabb15aa1f058e75f57b90abfc76a5f97500042.

Full public003 fixed-world source29-versus-measured23 video7177 is COMPLETE in
decoder_wide_lora_v1/regression_v1/cpu100_walk003. All1570 saved states rendered;
FFprobe confirms31.40s,1600x688,50fps,1570 decoded frames. First/middle/final
previews checked. Final root offset2.923m remains visible. SHA256
9ea0ec51584d089650d8312b162e18e6fdc39c1cb573a6bf296ec7190ce5e495.
No new physics or robot commands. Public recordings are TWIST2 PICO-workflow retargeted references,
not verified raw headset data. Same lossless002/003/dance training mixture and
optimizer-excluded development008; no broad generalization or hardware claim.

Disposition: REJECTED_NO_EXTENSION. Decoder-layer coverage alone under this cap
does not solve native23 path/contact control. No live export, launcher/interlock
change, physical movement, damping fix, commit or push. Goal ACTIVE; deployment
NOT READY. Live root estimation, real PICO/transport/dropout/handoff, smooth
control and fresh operator-supervised hardware evidence remain required.

## 2026-09-09,04:31 UTC (14:31 Sydney): decoder-wide original-intent LoRA implemented; smoke verified; capped100 ACTIVE

Previous goal turn was diagnostic progress, rejecting gains0/2. No scalar-gain
sweep follows. Source review confirms the current original-intent adapter only
touches the first affine, while the transfer paper describes decoder-layer
adapters. Existing older all-layer code used a different source/reference setup.
This restriction is an architectural fact, not a proved cause of control failure.

Implemented a distinct decoder-wide original-intent actor, runner, trainer,
weights-only CPU reader/recorder and saved-update audit. All nine low-latency
SiLU decoder affines receive rank16, scaling1 adapters:507792 LoRA parameters
plus36864 root weights; all28 released encoder/decoder tensors stay frozen.
Existing first A/B and source-token/masked930 history/root9 inputs are preserved.
Eight new tail A/B pairs use local seeds20260909+layer index and zero B. Output
remains23 physical motors. No gain, range, torque cap, source or live-export edit.

45 tests PASS (25 actual-weight/gradient/reader/snapshot plus20 launcher/header
guards); initial test run had two copied fixture references incorrectly treating
ParameterList as a single factor. Corrected the test references, preserved the
failed XML, then all tests pass. Ruff passes. First-layer-only reader rejects
the new actor instead of silently omitting tail adapters. Old exports reject it.

GPU smoke76713 completes2 updates/64 actual transitions. Audit34106 verifies all
18 adapter factors change, all10 encoder and18 base-decoder tensors stay exact,
root/exploration/critic update, finite optimizer and unchanged286-input critic.
Smoke audit SHA256 dc29b4a0b9c0b3b7342dd449b42b3aed30c9c8eddd590b7f39589870e9c9848b.

One fresh100-update32x64 regression is ACTIVE, session90621. At latest observed
iteration43/100 it has90112 transitions; do not restart on quiet output. Same
walk002/003/SONIC-dance training mixture, excluded-development008, original29
hand/head objectives, unclipped critic, actor clip0.2, physical limits/clock,
learning rates and source buffer as the first-affine unclipped comparison.
No resume or automatic extension. This is not broad-corpus parity yet.

Initial comparison3666 PASSES: all32 common actor tensors, all critic tensors,
all32 realized reset anchors and complete lifecycle arrays match first-affine0
bit-exactly. Additional16 factor tensors are the declared zero-effect tail.
All5396 saved decoder inputs give exact first-affine/all-layer initial means.
The full agent config changes only actor class and experiment name. SHA256
854511020cf434838f116210d3570623db0adb9af2cf9ec35f080d233f63f371.

Plan/artifacts: C:/Users/camer/sonic23_sim_artifacts/internet_pico_20260909_v1/
decoder_wide_lora_v1/. Pending: trained100 audit, four NEW complete CPU trials
through the unchanged range guard, then independent input/network/physics and
original-world/limb/standing comparisons. New failure wrapper preserves completed
physics PLUS all attempted adapter inputs/proposals, without trimming evidence.
No motion-quality result, policy promotion, physical robot action or damping fix
is claimed. Goal ACTIVE; full-body deployment remains NOT READY.

## 2026-09-09,03:57 UTC (13:57 Sydney): eight closed-loop root-gain attempts COMPLETE; both changes REJECTED

Previous goal turn made a nominal ankle-range correction. This turn is
DIAGNOSTIC PROGRESS ONLY: it rules out the two tested scalar root-position
changes, not a controller-quality win or deployment readiness. Exactly eight
new full-duration attempts run: gain0/gain2 on public002/003/008 and SONIC dance.
Gain1 explicitly reuses the four audited guarded baselines. No new training.

Only three root-position feedback features change; velocity, token, pose branch,
all checkpoint bytes, physical model, gains/caps, bounds, source samples/timing,
original29 hand/head intent, range preview and all screens stay fixed. The
actual and gained feedback are recorded separately. No live launcher changed.

Both gains stop002 before an unchecked target: gain0 completes525/1417 controls,
gain2 completes523/1417. Cause: bounded inward range-preview search finds no
verified next-control target. Last left ankle roll is−0.259885/−0.259244rad at
−0.447045/−0.941991rad/s; saved prefixes have zero range excess. A later legacy
recorder timing-shape exception does not erase those genuine guard stops.
Rescue files preserve actual physics; neither prefix counts as a full result.

The other six runs complete. Root q2 p95, gain0/gain1/gain2:
0032.263965/1.956218/2.579436m;0083.290986/3.111962/3.036690m;
dance2.079502/2.156039/2.114303m. Gain2 slightly improves008 and dance, worsens003,
and loses full002 completion. Final root offsets and limb metrics show mixed
effects, not full-body parity. All six world screens fail; standing/handoff
remain unqualified. Native dynamics/path tracking is still the main blocker.

30 focused tests and Ruff pass. Independent audit66330 binds278 inputs,
verifies9006 received root/token rows and8926 actual previous-action histories,
and reintegrates90060 saved substeps bit-exactly (79580 full-run,10480 failed-
prefix). Six full network traces reproduce within4.29154e-6 raw units and
1.90735e-6rad nominal target error. Original task/final standing metrics also
reproduce exactly. Failed rescue traces lack final model-output/preview fields;
their stronger policy-output qualification is explicitly not claimed.

First audit24317 stopped on Python tuple versus JSON list metadata, not numeric
motion disagreement. Numeric checker58281 confirmed exact JSON-normalized
equality; corrected separate audit_v2.py passes without rerunning any motion.
Both audit source versions and failure record are retained. Source inspection
also confirms an existing quadratic root-world penalty; absence of a position
reward or exponential saturation alone is not an established cause.

Outcome: C:/Users/camer/sonic23_sim_artifacts/internet_pico_20260909_v1/
root_gain_closed_loop_v1/OUTCOME.md. Audit SHA256
6eb2e1a8fd92f4a0b9cffb44ce0b4a06aea2f6c57d6325e3f91f0b17e43dceb3.
Experiment96488/tests87887/audit66330 terminal. No automatic gain sweep, extra
training, promotion, video, robot commands, interlock edits, commit or push.
Goal ACTIVE, deployment NOT READY. Needed next: a testable controller-level
change for sustained native23 whole-body/contact tracking; the tested scalar
changes do not establish that solution. Smooth braking, full50Hz timing, live
root estimation/PICO transport, physical damping and supervised evidence remain.

## 2026-09-09,03:26 UTC (13:26 Sydney): ankle-range repair verified in four full SIM motions; full-body tracking still fails

Previous goal turn was PROGRESS (controlled training/full-motion evidence).
This turn adds a measured controller correction, not deployment readiness.
First, 12 old saved-target physics replays reproduce qpos/qvel bit-exactly:
161880 substeps. All six ankle excursions have inward, unsaturated braking
opposed by contact loads. At unclipped002 first crossing, contact−2.723213Nm
opposes motor+2.723833Nm while ankle moves outward0.91849rad/s. Root drift is
separate; dance root-position branch changes lower13 targets only0.002575rad
RMS during motion. That fixed-input ablation does not prove a stronger-gain fix.

Implemented offline20ms joint-range preview using copied state and a separate
nominal model. Same source, actor bytes, gains, physical limits and gates.
Four fresh full CPU trials complete:0021417,0031569,0081114,dance1296 controls.
Only four total controls intervene (002524/525;008458/459). Actual range excess
0020.016999→0rad;0080.003493→0rad;003/dance stay0. Minimum actual margins on the
two corrected walks exceed1.9mrad. No midrun plant reset, source reanchor or
fallback. Previous-action history records the corrected physical target.

Root p95 before→guard:0020.665200→0.656198m;0031.956218 unchanged;
0083.412972→3.111962m;dance2.156039 unchanged. Leg12 RMSE worsens2.06%/1.10%
on002/008; final root maxima0.308937/1.950963/3.147855/2.341191m. All source-world
screens still fail; standing/handoff unqualified. Corrected prior OUTCOME's
leg-metric label from13 to12; diagnostic lower13 branch metrics include waist.

18 tests and Ruff pass. Independent audit77878 reintegrates all53960 new
substeps bit-exactly, checks223 inputs and5356 corrected-action history rows.
Predictions match these nominal states within2e-15. No hardware robustness or
recursive guarantee: max target correction0.302900rad,002 full-vector target
jump grows0.775621→1.057368rad; preview alone peaks17.18ms before inference/
transport. Keep offline; no promotion, tuning extension or hardware port.

Outcome: C:/Users/camer/sonic23_sim_artifacts/internet_pico_20260909_v1/
range_preview_v1/OUTCOME.md. Audit SHA256
e15c2fcc681249c4c938cee552878f6ce265d4374a9ab00c82b4f87a381e17c9.
Saved failure diagnosis: saved_failure_localization_v1/OUTCOME.md in same root.
Sessions70512/57674/84013/77878 terminal. No new training, robot commands,
interlock edits, video, commit or push. Goal ACTIVE; full-body deployment NOT
READY. Next: native world-path control and slew-constrained anticipatory braking;
physical damping, real headset/estimator and supervised hardware evidence remain.

## 2026-09-09,02:40 UTC (12:40 Sydney): unclipped critic100 COMPLETE; two paths improve, two regress — REJECTED

Supersedes ACTIVE entry below.100 updates/204800 transitions finish in session
10204 (11m42s reported training loop). Initial32 actor tensors/critic states,
all32 reset anchors and lifecycle arrays match clipped training exactly; entire
agent config differs only in use_clipped_value_loss=False. Same limits, rewards,
source timing, pose architecture and all28 frozen SONIC base tensors. Strict
checkpoint audit46695 passes;18 tests pass. No further updates or robot activity.

Four NEW full CPU replays complete in24326. Root p95 initial/clipped/unclipped:
walk0020.733112/0.863777/0.665200m; turn0032.073587/2.964635/1.956218m;
development0083.231815/3.384920/3.412972m; SONIC dance1.709359/1.807841/2.156039m.
Walk/turn paths improve23/34% against clipped100, also better than initial.
Development/dance worsen0.8/19.3%. Leg RMSE clipped→unclipped:
0.166238→0.165485 /0.194908→0.197609 /0.207327→0.205851 /0.217776→0.215311rad.
All four world/standing screens still fail. No overall deployment fix.

Physics audit64689 verifies107920 substeps (53960NEW),229 inputs, all actual
PD/effort clipping/ranges, packet/FK/phase and endpoint checks. Walk002 left ankle
range excess WORSENS0.010982→0.016999rad:−0.278799rad at10.548s versus−0.2618
lower bound,46/14170 post-substeps outside some range versus22 before.008 excess
improves0.007169→0.003493rad;003/dance0. Final standing root maxima
0.230885/1.950963/3.443684/2.341191m; joint maxima0.239686/0.256741/0.240773/
0.232432rad. Audit13936 final decision REJECTED_NO_EXTENSION.

Actual value-learning constraint was verified, but correcting it did not solve
full-body control. No assumption that critic clipping explains physical damping
or all path drift. No promotion, hardware commands, limit changes or training
extension. Next: explain motion-dependent path/leg trade-offs and ankle excursion
from saved traces before more training. Goal active; live teleop NOT READY.

Outcome: C:/Users/camer/sonic23_sim_artifacts/internet_pico_20260909_v1/
original_intent_unclipped_value_v1/regression_v2/OUTCOME.md.
Checkpoint100 d03684aec779b718333865c68fb50e23ff206f6658a65d2dd2bde61b532cbc74;
comparison3ef9cf8cd239c06791e23ab963fedf5841d4c6a577ee76de5aa6c72180c55429.
New walk002 fixed-world video renders every1418 states;50Hz,28.36s,1600x688.
First/mid/final inspected; source29 prescribed versus measured23 SIM, not robot.
Sessions10204/95960/46695/24326/64689/13936/45471 all terminal; no Python
experiment remains. Original repository and physical robot untouched.

## 2026-09-09,02:03 UTC (12:03 Sydney): actual value-learning diagnostic COMPLETE; one unclipped-critic test ACTIVE

Measured an optimizer constraint, not another assumed cause. Checkpoints0/100
each collected4096 actual fixed-policy stochastic simulator transitions with
unchanged source/reward/physics and stock input-normalizer updates. Same32
reset anchors. Independent audit24920 replays all stored predictions,
normalization, bootstrap/GAE and eight disposable critic fits;169 inputs checked.
No actor updates or hardware commands. Audit SHA256
95f31d3b715ef6f34add62837fd74eb84d816d9c3ce7a056eed47a15e4442e0e.

Trained-critic value clipping creates zero-gradient branches on58.40/60.94%
of minibatch samples. Same-data unclipped fits finish at658.813/653.434 MSE
versus943.988/839.282 clipped:30.2/22.1% less error. Explained variance was
already0.884/0.941, so this does NOT establish a useless critic or sole cause
of tracking failure. Actual timeouts0; timeout math tested synthetically only.
Captured full reward inventory; feet-position cost dominates initial-block
average penalties. No reward redesign or threshold change authorized by this.

New additive driver train_g1_true23_pose_unclipped_value.py changes only
algorithm.use_clipped_value_loss=False. Same pose actor/snapshot schema and
offline inference, but distinct full training lineage with pinned diagnostic
proof. Policy clip0.2, critic gradient norm0.5, learning rates, all physical
limits, motions, observation timing and rewards unchanged.18 tests PASS in
33.78s; Ruff PASS. One fresh100-update32x64 regression launched as session10204,
not a continuation of rejected pose100. Initial-state equivalence, strict update
audit and the same four complete CPU replays remain required. No claim that
lower critic fitting loss fixes motion. Hardware/teleop remains NOT READY.

Diagnostic outcome and captures: C:/Users/camer/sonic23_sim_artifacts/
internet_pico_20260909_v1/original_intent_value_diagnostic_v1/OUTCOME.md.
Candidate: original_intent_unclipped_value_v1/EXPERIMENT.md and regression_v2.
Setup29245 stopped before simulator construction because the new guard checked
max_grad_norm before the common CLI sets0.5. Guard ordering corrected and a real
factory test added;18 tests PASS in30.08s. No prior training updates to resume.
Sessions20565/17332/24920 all terminal. Prior failed diagnostic setup83485
performed no rollout; its driver API error was corrected before new paths.

## 2026-09-09,01:21 UTC (11:21 Sydney): pose-conditioned100 COMPLETE; four full replays, all path errors worse than initial — REJECTED

Supersedes ACTIVE entry below. Session74492 training completed100 updates /
204800 transitions in13m59s; session14511 strict update audit PASS. All10 encoder
and18 decoder tensors remain bit-exact; both pose factors, root/noise/critic
updated. Pose conditioning works at zero root input but does not solve tracking.
No training extension, resume, gain/limit/physics/source-time or objective change.

| Complete source | Root p95:initial / prior mixed-root100 / pose100 | Leg RMSE:initial→pose100 | Hard-range excess:initial→pose100 |
| --- | --- | --- | --- |
| Walk002:1417 lifecycle/667 source controls |0.733112 /0.842913 /0.863777m|0.168435→0.166238rad|0.008217→0.010982rad|
| Turn003:1569/819 |2.073587 /3.138591 /2.964635m|0.191097→0.194908rad|0→0rad|
| Walk008:1114/364 |3.231815 /3.650319 /3.384920m|0.213216→0.207327rad|0.009802→0.007169rad|
| Original SONIC dance:1296/546 |1.709359 /1.746426 /1.807841m|0.225615→0.217776rad|0→0rad|

All finish without the fall stop; all unchanged world/standing screens fail.
Root errors worsen17.8/43.0/4.7/5.8% versus initial. Leg errors improve on three
motions; turn leg error worsens. Turning/walk008 path improves versus the prior
rejected root-only candidate, while walk002/dance worsens. No overall fix or
promotion. Final standing root maxima0.274857/3.071515/3.439793/1.921973m and
joint maxima0.239486/0.238945/0.229912/0.249967rad remain unqualified.

40 actual-weight tests PASS; scoped Ruff PASS. Initial common30 tensors and149
sampled means across all4 baseline traces match exactly; new branch has exactly
zero initial effect. All4 baselines reused explicitly. Initial critic tensors
and all32 reset anchors match previous mixed training; actual curriculum matches
all4315 prepared lifecycle frames. Equal realized clip exposure is NOT claimed.
Session87981 completes4 NEW CPU attempts,53960 new substeps. Session75352 audits
107920 saved substeps including reuse,228 bound inputs, all PD/engine clips,
actual ranges, received inputs, original29 FK, q1/q2 and lifecycle endpoints.
Session29138 matched comparison passes; candidate decision REJECTED. All these
sessions are terminal; final process check finds no remaining Python experiment.

Outcome: C:/Users/camer/sonic23_sim_artifacts/internet_pico_20260909_v1/
original_intent_pose_lora_v1/regression_v1/OUTCOME.md. Checkpoint100 SHA256
e9f4147d0b0a412f457fcc4407ccf4cf7c877b43613208d17395d34f654c606a;
comparison99925acb63b4630a040fe96750b6fe360915723c6d5ec491bf04a3e16d279f56.
This is first-affine LoRA, not the original all-layer transfer recipe. Do not
extend this rejected run. Before more policy training, inspect actual on-policy
return/value calibration and value-clipping behavior; logged value loss alone
does not establish a critic defect. No such change or fix is claimed here.

No new video or relabelled old replay; no hardware/DDS/arming/mode/recovery,
export, promotion, commit or push. Existing physical-world estimator, live
headset/dropout/handoff and prior motor damping remain unqualified. Goal ACTIVE,
incomplete. Preserve completed source closures; don't rerun these experiments.

## 2026-09-09,00:50 UTC (10:50 Sydney): pose-conditioned frozen-base adapter implemented; smoke verified, fresh100 ACTIVE

Previous goal turn classified PROGRESS: the mixed-data root-only implementation
and four complete CPU comparisons produced a verified rejection, not readiness.
Current installed RSL-RL PPO clips actor and critic gradients separately, ruling
out the proposed shared-gradient-clipping starvation mechanism without another
experiment. Root-only adaptation is exactly zero at zero9 feedback; frozen SONIC
still sees pose intent, but that trainable branch cannot change this response.
This limitation is not by itself proof of the failures' full cause.

New separate native23_pose_lora actor/trainer/runner/CPU reader/auditor adds a
rank16 994→16→4096 first-affine correction (81440 parameters) beside root9.
Inputs remain frozen source64 token plus masked930 physical history, no extra
privileged observations or invented absent-axis feedback. All released base
tensors remain frozen. Locally seeded A and zero B preserve initial means and
do not consume extra global RNG; this is not the original all-layer LoRA recipe.
Old snapshots/exports stay incompatible. No old hash-bound source was edited.

40 focused real-weight tests PASS in87.05s. Tests cover zero-effect initial
means, actual zero-root learning, frozen-base gradients/tamper guards, exact
optimizer ownership/counters and CPU inference with a nonzero pose branch.
The CPU reader was explicitly implemented to avoid silently calling the legacy
root-only mean. Scoped Ruff passes. Fresh2-update smoke completes64 transitions;
strict audit verifies all28 base tensors fixed, both pose factors/root/noise/
critic updated, finite optimizer and286-input critic, and old exporter rejection.
Smoke audit SHA256 8904d1545ea52eeb90112ead1096574482a2835e4554df6f752ff5de65edf2d8.

Fresh100 trial is ACTIVE, terminal session74492, under
C:/Users/camer/sonic23_sim_artifacts/internet_pico_20260909_v1/
original_intent_pose_lora_v1/regression_v1/. Same unchanged three-recording
dataset from original_intent_multimotion_v1, same32×64×100=204800 transitions,
two epochs/four minibatches, source timing, physics, rewards and physical gates.
Only the additional pose adapter has a new fixed1e-4 learning-rate group.
No resume/extension. Four complete CPU attempts and independently audited
physics are prepared, not executed yet. All four old initial traces require
exact common-state/zero-adapter and sampled-mean proof before reuse. Do not edit
the executed source closure while this process or its future evidence is in use.
No new motion improvement claimed. No robot/DDS/mode/arming/recovery/deployment.
Goal remains ACTIVE; full-body teleop/hardware handoff/damping are unqualified.

## 2026-09-09,00:26 UTC (10:26 Sydney): three-recording100 experiment COMPLETE; all four path errors worsen — REJECTED

Supersedes ACTIVE status below: session53109 training, all six new CPU
evaluations, update/equivalence/physics/comparison audits and session29657 full
dance rendering are terminal. No experiment process remains running.

Lossless publicwalk002 + turn003 + original SONIC dance mix trained fresh100
updates/204800 total transitions in12m31s. Same frozen-base/root-only actor;
all10 encoder and18 decoder tensors remain bit-exact. No target scaling,
retiming, gain/limit/physics change or old checkpoint continuation. Walk008 is
optimizer-excluded DEVELOPMENT, not untouched heldout data. Public references
come from TWIST2's PICO workflow, not authenticated raw headset captures.

| Full source | Root p95:initial→mixed100 | Leg RMSE | Actual hard-range excess |
| --- | --- | --- | --- |
| Walk002:1417 lifecycle/667 source controls |0.733112→0.842913m|0.168435→0.181210rad|0.008217→0.011062rad|
| Turn003:1569/819 |2.073587→3.138591m|0.191097→0.186773rad|0→0rad|
| Walk008:1114/364 |3.231815→3.650319m|0.213216→0.216316rad|0.009802→0.006390rad|
| Original happy dance:1296/546 |1.709359→1.746426m|0.225615→0.227069rad|0→0rad|

All complete without a fall; all unchanged world/standing screens still fail.
Root errors worsen15.0/51.4/12.9/2.2%. Turning leg error and walk008 range excess
improve, and some standing metrics improve; none qualifies full-body tracking.
Candidate REJECTED, no extension. More data alone at this fixed budget did not
fix this root-conditioner architecture; not proof all multi-motion training
cannot work. Do not repeat this capped run or relax physical/travel criteria.

57 dataset/reference/environment tests PASS with no skips; scoped Ruff PASS.
Prepared4315 lifecycle frames match actual training curriculum bit-for-bit.
Initial all30 tensors and82 saved CPU probes prove exact baseline reuse for
002/003 only; initial008/dance and all trained cases are NEW complete CPU runs.
Independent audit verifies107920 saved substeps,78060 NEW,221 bound inputs,
every PD/engine clip, actual joint ranges, received inputs, original29 FK and
q1/q2 metrics. All four paired source/setup fields match. Validity PASS is not
motion qualification. Dataset assembly and horizon boundaries are tested.

Completed outcome: C:/Users/camer/sonic23_sim_artifacts/internet_pico_20260909_v1/
original_intent_multimotion_v1/regression_v1/OUTCOME.md. Comparison SHA256
5075dd83b682760f52aa02768d22c05352947d63ac786384035d7ae6a3747ddc; checkpoint100
79033df41dc6c75ce15d48ddc3963eca35b1eaa01718b118532b0f6150598ff1.
Full fixed-world dance video: cpu100_dance/source_vs_measured.fixed-world-v3.mp4,
1297 frames/50Hz/25.94s/1600x688, ffprobe and first/middle/final inspection pass.
Original29 prescribed poses left, measurednative23 SIM right; path drift is
visible. Video SHA256
9449ebdc1b59685268637d208ea9e65168f0514ec342541ff0e7179f7caa21e6.

No physical/DDS/LowCmd/arming/mode/recovery, promotion, live export, commit or
push. Do not edit or rerun the completed experiment's hash-bound source closure.
Live root estimation, headset/dropout/handoff and prior hardware damping remain
unqualified. Goal ACTIVE and incomplete; live full-body teleop is NOT ready.

## 2026-09-08,23:56 UTC (Sep9,09:56 Sydney): lossless three-recording mix implemented; frozen-base100 training ACTIVE

Current continuation made progress; no deployment claim. Rechecked the other
`23dofsonic` task and original deployment-observation work. Its historical
"passed" walks were survival screens with world errors above2m, not full-body
qualification. Its exact policy was already compared on public002/003/008 in
the15:28 entry below; C++ observation differences were already measured Sep6.
Those are not new discoveries or reasons to repeat those experiments.

New explicit data hypothesis addresses the prior single-walk overfit: one
frozen-base root conditioner, uniform reset sampling of publicwalk002 (667
source frames), turn003 (819), original SONIC planned happy dance (546).
Walk008 remains optimizer-excluded development evaluation, NOT an untouched
test. The public files are PICO-workflow robot references, not authenticated
raw PICO tracker recordings. No target scaling, retiming, missing-axis erasure,
gain/limit change or physics modification. No old checkpoint continuation.

Added `g1_true23_original_intent_mix.py`, preparation CLI, and19 tests. Assembler
rejects duplicate recordings/content, partial/retimed clips and evaluation-ID
leakage. Each saved full lifecycle is independently regenerated before joining:
all2032 source frames,4315 lifecycle frames and original29 task arrays match
bit-for-bit. Spans are0:1428,1428:3008,3008:4315; horizons cannot read across
them.57 reference/environment tests PASS; scoped Ruff PASS.

New artifact root: `C:/Users/camer/sonic23_sim_artifacts/internet_pico_20260909_v1/`
`original_intent_multimotion_v1/`. Dataset report SHA256
`2eadae368ff9beecd803be96d07590608950840c586dbc0455fca3941672445c`.
Fresh smoke2/4env/8steps completes64 transitions; strict update audit PASS:
all28 base tensors fixed, root/noise/critic updated, source-aware critic286,
finite optimizer, correct counters and legacy exporter rejection.

Fresh regression100/32env/64steps/2epochs/4minibatches is ACTIVE (terminal
session53109); same204800 total-transition cap as the single-clip experiment,
not204800 perclip. No guaranteed equal realized clip exposure. Initial32-env
received-input parity covers all3 recordings, maximum encoder error5.6811e-8,
original world-target component error8.2119e-7m. At this entry,31 updates
reported63488 transitions; not finished and no policy outcome yet.

Predeclared next: strict100-update audit, initial-actor equivalence, six new
complete CPU attempts (newinitial008/dance plus trainedall4); existinginitial
002/003 reusable only after exact proof. Generic explicit-reference physics
auditor and four-case matched comparison are prepared. Do not rerun or extend
training after100. No hardware/DDS/LowCmd/arming/mode/recovery, promotion,
commit or push. Goal remains active; live teleop is NOT ready.

## 2026-09-08,23:17 UTC (Sep9,09:17 Sydney): frozen-decoder100 completes; walk improves, turning/physical limits regress — REJECTED

Fresh frozen-base experiment is finished:100 updates/204800 transitions,32
environments,64 rollout steps,2 epochs/4 minibatches,11m43s actual training.
All10 encoder and18 decoder tensors are bit-exact; all36864 root weights change,
critic/optimizer finite and strictly root/exploration/critic-only. Old readers
and exports reject the distinct research snapshot.130 focused tests PASS.

| Full public source / same PyTorch CPU | Root p95:initial→frozen100 | Leg RMSE | Hard joint-range excess |
| --- | --- | --- | --- |
| Walk002:1417 lifecycle/667 source controls |0.733112→0.628548m|0.168435→0.170090rad|0.008217→0.013618rad|
| Turn003:1569 lifecycle/819 source controls |2.073587→2.762237m|0.191097→0.194189rad|0→0.007359rad|

Both complete upright but all world/standing screens fail. Walk root improves
14.3%; turning worsens33.2%. Both worst physical excursions are left ankle roll
during source motion, despite in-range positive requested angles. Final standing
root errors grow to0.319285m/2.811444m. Frozen decoder solves the weight-drift
mechanism, NOT transfer or physical qualification. Candidate rejected; no more
updates, repeat trial, gain sweep or gate relaxation for this run.

Initial CPU baselines reused only after all30 tensors, identical mean/token/codec
methods and82 saved-input probes verify bit-exact means. Independent physics
audit checks59720 saved substeps (29860 new),195 inputs, all PD/engine clips,
actual joint limits, received inputs, original29 task FK, q1/q2 metrics and full
lifecycle. Matched comparison verifies identical physical/source/setup fields.
This turn's separate component isolation added31380 other new substeps; do not
count reused initial/full100 traces as fresh simulations.

Outcome: `C:/Users/camer/sonic23_sim_artifacts/internet_pico_20260909_v1/`
`original_intent_frozen_decoder_v1/regression_v1/OUTCOME.md`. Comparison SHA256
`1ca663d7700f04f53c39dabb2844d1eba50bfd5167c35ffaa27b25134167ebfa`; checkpoint100
`d7859252b23085e7cd77d5bc923a720d43bb1a9f0280a35e26e031ff4cc6bc02`.
Fixed-world original29 prescribed versus measured23 turning video retains all
1570 saved states,50Hz/31.40s,1600x688; checked with ffprobe and preview inspection.
Video SHA256 `2d33235317dc2a8dead90c6f228ae60e279fd5ccf275a6d043ae22f50093e425`.
Earlier v1 initial-trace display had duplicate-floor z-fighting; rejected visual
retained separately. No physics/checkpoint/source change was made for rendering.

All experiment processes finished. No hardware/DDS/LowCmd/arming/mode/recovery,
promotion, commit or push. Physical damping cause and live root estimation,
headset transport/dropout/handoff remain unresolved. Broad goal stays active,
not deployment-ready. Next work must address transfer/dynamic constraints with
new evidence, not extend this single-clip seed or relabel a surviving replay.

## 2026-09-08,22:50 UTC (Sep9,08:50 Sydney): decoder drift isolated; frozen-base GPU smoke verified, fresh100 initializing

Completed exactly two new full turning component trials, all1,569 lifecycle/
819 source controls, no fall or hard joint-range excess. Same PyTorch CPU
physics/source/initial state/gains/limits as the existing initial/full100 trials.
Root p95: initial2.073587m, conditioner-only1.982513m, decoder-only2.374621m,
full1002.643661m. Conditioner-only leg RMSE0.187387rad versus initial0.191097rad,
but final standing joint error worsens0.210946→0.218165rad. All world/standing
screens still fail. No composite/checkpoint is selected or promoted.

Independent reconstruction verifies both composite actor digests, exact parent
weight partitions, unchanged setup and all received source arrays. Physics
audit rechecks62,760 saved substeps (31,380 new); shared-action audit reproduces
all1,196 sampled saved actions bit-exactly. Decoder movement dominates the root
branch at these inputs; decoder-only closed-loop turning regresses14.5%.
Completed outcome: `C:/Users/camer/sonic23_sim_artifacts/internet_pico_20260909_v1/`
`original_intent_component_ablation_v1/OUTCOME.md`. Comparison SHA256
`d1e684958d89056a2dbec36fed6d450da9ccf5491e39665cd1ce8208b4c6e4cc`.

Implemented a distinct frozen-decoder actor/runner and fresh-only trainer. The
encoder AND all18 decoder tensors are pinned; only root conditioner, bounded
exploration and fresh critic are optimized. Gradients traverse the frozen MLP
without accumulating on it. Snapshot header/actor kind/optimizer partition are
different; old reader/export acceptance and resume are forbidden. Actual source
targets, objective, critic286,200ms timing, plant and physical limits unchanged.
Existing executed source files/checkpoints are not edited or relabelled.

The combined130-test suite PASS in130.91s, no skips:37 new frozen-base tests plus
93 existing source/environment/CPU-contract tests. The two-update GPU smoke
completes64 actual transitions; all10 encoder and18 decoder tensors unchanged,
all36,864 root weights updated, critic/optimizer finite, real286-value critic,
old export rejection and actual nonzero-feedback action changes verified.
Smoke audit SHA256 `7b6e2cf8719590a69994ea3bfa9dc08a4189b333c08289e577af5abad0d05404`.
A separate fresh capped100-update regression is initializing; no full motion
result is claimed for this frozen-decoder recipe yet.
No extra updates to the rejected full-decoder seed. Immutable plan is
`original_intent_frozen_decoder_v1/EXPERIMENT.md` under the public-motion artifacts.
No robot/DDS/LowCmd/arming/mode/recovery, live promotion, limit change or damping
fix. Broad full-body teleop goal remains active/incomplete, not deployment-ready.

## 2026-09-08, 22:06 UTC (Sep 9, 08:06 Sydney):100-update source-aware regression completes; full motions survive but candidate REJECTED

The capped fresh original-intent regression is finished, not running. Exactly
100 updates/204,800 transitions,32 environments,64 steps/environment/update,
2 PPO epochs and4 minibatches;12m36s actual training time. Saved-state audit
verifies all18 decoder tensors changed, all10 encoder tensors frozen, all36,864
conditioner elements updated, finite critic/optimizer and exact source lineage.
New actor remains rejected by the old exporter/live contract. No extension,
physical commands, gain/limit changes, candidate promotion, commit or push.

| Complete public source / same PyTorch CPU backend | Root p95:0→100 | Leg RMSE:0→100 | Hard joint-range excess:0→100 |
| --- | --- | --- | --- |
| Walk002;1417 lifecycle /667 source controls |0.733112→0.711775m|0.168435→0.182407rad|0.008217→0.020089rad|
| Turn003;1569 lifecycle /819 source controls |2.073587→2.643661m|0.191097→0.199676rad|0→0rad|

All four comparison rows complete every source and entry/return control without
fall or command/engine force discrepancy. All original world/standing screens
still fail. Walk root improves2.9% but leg error and physical range excess worsen;
turn root worsens27.5%. Final trained standing errors remain0.259rad/0.272m on
walk and0.225rad/2.662m on turn. Original hand/head task errors do not show a
general improvement either. Candidate fails the predeclared selection criteria.
It is rejected and this run is not extended. Surviving is not deployment-ready.

Initial walking trace is reused without relabelling: all30 smoke0/regression0
actor tensors and actor contracts are bit-exact. Three new regression CPU
attempts plus the existing initial walk form the comparison. Independent audit
checks59,720 saved substeps, target transforms, PD/engine clips, received inputs,
original task FK/metrics and189 pinned inputs. The smoke comparison earlier in
this turn overlaps that initial walk; do not double-count it. Across the five
unique new CPU attempts in this training work,73,890 substeps were integrated.

Full original-source intent is now part of the training task, unlike the old
native-upper-posture objective. This implementation is verified, but the first
single-clip training result is not a generalized controller fix.93 focused tests
PASS in80.13s; scoped Ruff and document diff checks pass. No deployment exporter
or physical transport accepts this new actor. Same-input old ONNX/new PyTorch
check on all1417 saved walking inputs has identical frozen tokens and maximum
raw action difference2.623e-6; closed-loop backend equivalence is not claimed.

Evidence: `C:/Users/camer/sonic23_sim_artifacts/internet_pico_20260909_v1/`
`original_intent_training_v1/regression_v1/OUTCOME.md` and immutable experiment.
Checkpoint100 SHA256 `8ef0b4bd694566e78dca17789c701cfad7c277891255d6e00fb6d25134b62ed4`;
update audit `89ae5b0e1f7990c75a628a2c967863bdd76470186436ceb9cac02e85614e8094`;
comparison `039648e800808cbd35efdba2e1c726b1a91ca29ea448f5740144941fb39cb9cc`;
initial equivalence/backend audit
`22965ff94cea2c59c9624c5051426e5d95761fd5c0930a1a91d61586b2971c81`.

No training/evaluation process from this experiment remains live. Next work
must address the failing native23 controller, not extend this rejected seed or
relax original travel/physical gates. Source/task correctness and rewards alone
have not established dynamic feasibility or full-body parity. Headset transport,
root estimation, physical handoff and previous motor damping remain unqualified.
The scaled-travel question is unanswered. Broader goal remains active/incomplete.

## 2026-09-08, 21:43 UTC (Sep 9, 07:43 Sydney): original-intent training runs; smoke verified, full walking accuracy still FAIL

Implemented a distinct `native23_original_source_tasks_v1` training recipe.
All29 source joints contribute to original hand/head VR and world task targets;
the physical model, measured joint feedback and commanded outputs remain23.
Original hand/head errors now receive explicit position/orientation costs.
Conflicting native upper-body/joint imitation is removed; native lower-body,
root and feet objectives remain. Existing numerical end-effector height gates
are preserved while their hand targets use the explicitly declared neutral
wrist proxy. No physical gain, effort, joint-range or acceptance gate changes.
The value network additionally sees original received-q1 VR21 (286 total critic
inputs); actor ABI remains267 encoder,994+9 decoder,23 output. This is not a
relabelled legacy checkpoint or live-compatible export.

First GPU smoke stopped before any PPO update/checkpoint: startup receipt
expected `threshold` metadata omitted by the new termination. Fixed that while
retaining0.25m, preserved the failed source snapshot, and retried in `smoke_v2`.
Retry completes exactly2 updates. Actual smoke cap is8 steps/environment,
4 environments,2 epochs and2 minibatches:64 transitions, not the requested256.
Independent saved-checkpoint audit verifies all18 decoder tensors changed,
all10 encoder tensors frozen, conditioner/noise updates, finite critic/optimizer,
resolved counters, exact initial released weights and rejection by old exporter.

Two matched full PyTorch-CPU walking tests finish all1,417 lifecycle controls
and all667 original public002 source controls, including standing return.
The smoke does not improve tracking and is not a deployment candidate:

| Same CPU backend / original-source targets | Root-position p95 | Leg RMSE | Hard joint-range excess | World screen |
| --- | --- | --- | --- | --- |
| Checkpoint0 |0.733112m|0.168435rad|0.008217rad|FAIL|
| Checkpoint2 |0.796250m|0.170082rad|0.017910rad|FAIL|

All28,340 saved substeps, physical target transforms, requested PD/engine clips,
received inputs and original task geometry are independently rechecked. No
state discontinuity, command/engine disagreement or fall. Original task metrics
are reported at both held received-q1 and the unchanged referee's post-control
q2 source index; q1 does not replace existing q2 gates. Hand/head pelvis-centered
position p95 changes0.154/0.156/0.141 to0.151/0.151/0.136m, while world drift
worsens. Final standing position/joint errors remain excessive. PyTorch zero
differs from the earlier ONNX result; comparisons here use the same backend.

New additive tools: `verify_g1_true23_original_intent_update.py`, strict offline
`g1_true23_original_intent_checkpoint.py`, and full lifecycle recorder
`record_g1_true23_original_intent.py`. The reader verifies actor/objective/source
lineage and cannot authorize the legacy live transport/export. Combined93
objective/reference/CPU-reader tests PASS in80.13s, with no skips. Scoped Ruff
passes. Independent smoke comparison SHA256 is
`1c4ff898077c248c7f94d8a937844122603a4c8d466bf811ba7ad67627b80523`.

One fresh capped100-update/32-env regression is now running, with explicit
`regression_v1/EXPERIMENT.md`, same new source/task recipe and no auxiliary
projection imitation. Requested64 steps/environment/update; actual resolved
batch/counters will be checked. It starts fresh from the released source, not
smoke2 or the previously rejected old-objective training. After audit: exactly
four full CPU cases, checkpoint0/100 on public002 and turning003. Stop at100;
no automatic extension or promotion based on reward trend. This run is not yet
complete and no tracking improvement is claimed.

Evidence: `C:/Users/camer/sonic23_sim_artifacts/internet_pico_20260909_v1/` under
`original_intent_training_v1/`. Smoke update audit SHA256
`fa48dc0caf6ca23d6d439693f071ee4f5d25c55f64e781689aad85f90650ded6`;
`smoke_v2/comparison.json` contains the independent physics/task audit.
No hardware/DDS/mode commands, estimator or damping fix, changed live commands,
commit or push. Scaled-travel question is unanswered; original criteria stand.
Goal remains active; live full-body teleop is not qualified.

## 2026-09-08, 20:45 UTC (Sep 9, 06:45 Sydney): full original dance completes; lossless source-intent preparation implemented; tracking remains FAIL

This goal turn is PROGRESS, not deployment qualification. Three new full-speed
original SONIC `happy_dance` CPU attempts complete all 546 source controls and
the same standing/entry/return timeline: 1,296 controls each. No falls, physical
joint-range excess, or engine-versus-clipped-command discrepancy in these three
attempts. No training, hardware/DDS, mode commands, limit/gain changes, promotion,
commit or push. The earlier public002 ankle overshoot remains unresolved.

| Variant | Lifecycle controls | Full-source root-position p95 | Leg-joint RMSE | World tracking |
| --- | --- | --- | --- | --- |
| Native23 zero-update; absent source axes zeroed in VR | 1296/1296 | 1.457m | 0.171rad | FAIL |
| Same native23 actor; original29 hand/head VR preserved | 1296/1296 | 1.683m | 0.225rad | FAIL |
| Untouched original29 low-latency SONIC; all axes actuated | 1296/1296 | 1.536m | 0.157rad | FAIL |

The two native cases isolate VR21; all weights, physics, history, non-VR source
channels, and timing match. Inputs first differ at reference frame367/control358;
all prior states are bit-exact. Original29 also differs in model, gains, action
history handling, available axes and absence of the added root-feedback input;
it is not a morphology-only or full C++/firmware comparison. These are planned
source poses, not replayed executed robot states. Source speed remains1.0.

Preserved-source native23 ends upright, but final proof-window errors remain
0.239666rad maximum joint error and1.846114m root-position error. Its measured
root height stays above0.649775m; maximum tilt0.460220rad. Root drift is not the
only problem: original hand/head errors after subtracting each pelvis position
(world axes retained) have p95 0.341/0.360/0.249m; world orientation errors are
1.367/1.378/0.688rad. Native hands use the explicitly derived neutral-wrist proxy,
not a claim of identical physical hand/contact geometry. The all29 source has
smaller but still material body-relative position errors0.198/0.239/0.108m and
orientation errors1.270/1.266/0.339rad. Preserving intent helped public turn
survival previously, but worsens this dance's path/leg accuracy: not a universal
standalone controller fix. No original acceptance criterion is relaxed.

Implemented additive `g1_true23_original29_reference.py` and
`prepare_g1_true23_twist2_original_reference.py`: preserve all29 source axes for
hand/head FK, while physical feedback/actions remain23. The versioned bundle
retains source poses, VR21 and original world task positions/WXYZ, rejects bad
quaternions/model layouts and mismatched native pairs, and never relabels the
old zero-absent-reference checkpoints. It does not install a training reward or
live receiver. Prepared all three public sources directly, without needing an
original29 policy rollout. All4,133 lifecycle reference frames and4,100 received
controller inputs match the earlier executed full-source-VR experiments exactly.
Independent FK differs by at most8.88e-16m; world targets reconstructed from
serialized received input differ by at most2.49e-7m per position component.

Executed existing training reward functions on all1,850 public source poses.
The native-reference joint/body rewards reach their exact maximum1.0 and measured
posture costs0, while original hand targets still miss by roughly11–23cm p95.
Zeroing the six absent source axes also changes hand orientation by26–46degrees
p95. This proves a native-reference/original-intent objective mismatch, not that
it alone causes observed drift/falls. Existing reward profiles remain unchanged.

Checked the two upstream SONIC local VR observation formulas at git revision
`4141c34280abb67c82e115342a8720f4a83d750d`, with the same upstream repository's
WXYZ tensor math substituted for unavailable IsaacLab. Across all5,440 public
and dance lifecycle reference frames, maximum float32 position component error
is8.40e-7m and signed quaternion component error2.39e-7. Negative XYZW-order
checks fail as expected. Reference pelvis, not measured robot pelvis, is the
VR normalization frame. No axis/order/frame discrepancy found in this check;
this is a formula check, not exact IsaacLab or full C++ runtime execution.

Preflight caught tiny source WXYZ norm roundoff before any dance controller ran.
Explicit canonicalization before both lifecycles preserves every root XYZ/joint
value and represented rotation (maximum difference5.56e-17rad); raw546 poses are
retained beside canonical poses. No pair-matching tolerance was loosened.
Independent dance audit verifies38,880 substep state transitions/PD commands,
original source FK/metrics, native received inputs and matched-pair invariants.
49 focused repository tests PASS (27.99s), no skips; scoped Ruff PASS.

Evidence root: `C:/Users/camer/sonic23_sim_artifacts/internet_pico_20260909_v1/`.

- `original_source_bundle_audit_v1.json`: SHA256
  `ac99bb91ff7ade0e0a44a808e92a61f898f958d80fc5ae624dc32f5fb47a7547`.
- `original_intent_reward_mismatch_v1.json`: SHA256
  `d7e1a5d6294136b905b1b8e761baed8b2de85d4ed6a3fb4486a980645169828d`.
- `upstream_original_vr_convention_v1.json`: SHA256
  `a5f63089108f63f781d5bbc24af0f609cd8c41aff8762f348c2381edb3b98e35`.
- `sonic_happy_dance_original_vr_v1/comparison.json`: SHA256
  `a9c71f7d33c4df93fbf027bd02b340d946cf59ab1c96982f4d0710f3d8e3cf97`.
- Same dance folder: `OUTCOME.md`, immutable `EXPERIMENT.md`, three traces,
  and `native_original/nominal.measured-simulation.mp4`, SHA256
  `7598ec5f4bc4ecd6b7908259b85c166b44f71406dad1e2518017d28e67c8584a`.
  Video is1,297 saved-state frames/25.94s at50Hz (integrated time25.92s).
  Dance14s and final25s previews visually inspected. Following camera hides
  world-path error; video is not robot footage or tracking qualification.

Next controller experiment must explicitly couple preserved original task intent
to native23 feasible targets and training objectives at the same received frame,
with unchanged measured23 history, encoder ABI and physical limits. Reusing the
old native-only reward or extending the rejected walking fine-tune is not this
fix. Source hand/head, feet/root and standing must all be measured afterward.
Question about acceptable scaled travel is unanswered; no new acceptance rule
is authorized. Live estimator/transport, safe mode return and physical damping
remain unqualified. Goal stays active; no physical test is authorized by this SIM.

## 2026-09-08, 19:43 UTC (Sep 9, 05:43 Sydney): preserving original public hand/head intent removes the observed zero-update turning fall; full-body tracking still fails

This goal turn is PROGRESS, not deployment qualification. Seven new complete
nominal CPU attempts distinguish source-side SONIC drift from a native23 input
effect. No training, physical robot, DDS, mode transition, gain/effort change,
policy promotion, commit or push. Existing dirty work is preserved. These public
TWIST2/PICO-workflow recordings are development examples, not raw authenticated
headset packets or held-out generalization data.

New reusable `gear_sonic/scripts/record_g1_sonic_public29_baseline.py` retains
all29 source joints and runs untouched low-latency SONIC weights in the saved
original29 deployment scene with captured C++ command parameters. The complete
standing-entry-source-return-standing lifecycle, source speed, 50-Hz samples,
500-Hz physics and received200-ms horizon match the native23 comparison. Initial
retained joint angles match exactly. No added9-value root-feedback controller,
state rewrite or fallback. It is not complete C++/firmware reproduction or a
morphology-only ablation: model, gains, available axes, previous-action handling
and feedback differ. The source policy does not consume absolute root XY.

All29 completes all three lifecycles:0021417/1417,0031569/1569,0081114/1114.
Full-source root-position p95 is0.581/1.768/2.470m; leg-joint RMSE is
0.150/0.137/0.164rad. All world-landmark screens FAIL. On003 the original model
has0.024496rad waist-pitch range excess for79 substeps and its engine caps eight
left-hip-pitch commands at88Nm (largest requested92.668Nm), despite the legacy
outer clip allowing139Nm. No native23 limit was changed. Completion therefore
does not establish even original29 physical/tracking qualification.

Independent read-only audit reconstructs all4,100 measured histories and41,000
substep target/PD commands, state continuity, received inputs and source FK.
All histories/targets/PD commands are bit-exact; lower-body240 and timestamps
match paired native23 traces. Source/world and body-relative errors are both
reported; relative metrics do not replace the original world-space gates.

The remaining reference difference was isolated: previous native23 generation
zeroes the six absent source axes before computing hand/head VR21. A bounded
two-run003 experiment restores **original all29 VR21 only**, using each existing
native23 actor's unchanged timeline, physical model, gains, limits, action codec,
root feedback and weights. Full original FK, retained joints/root/frame counts,
and all non-VR source fields are independently checked. This intentionally
changes the checkpoint's training reference distribution; no manifest is
relabelled. Inputs first differ at source frame261 / encoder control252; every
earlier qpos/qvel state remains bit-exact against its prior run.

Zero-update003 now completes1,569/1,569 controls and all819 source controls,
versus the prior fall at1,068/1,569. Minimum physical root height0.694799m,
maximum tilt0.386116rad, no hard joint-range excess. Follow-up002 and008 also
complete, with the identical zero-update weights and original-source VR:

| Public clip | Full lifecycle / source controls | Full-source root-position p95 | Leg-joint RMSE | Maximum hard joint-range excess | World tracking |
| --- | --- | --- | --- | --- | --- |
| 002 |1417/1417;667/667|0.576m|0.169rad|0.008696rad|FAIL|
| 003 |1569/1569;819/819|1.832m|0.191rad|0|FAIL|
| 008 |1114/1114;364/364|2.915m|0.205rad|0|FAIL|

Do not mistake survival for a complete fix. Matched old→original-VR root p95:
0020.651→0.576m;0031.329→1.455m over the common718 source controls;
0082.554→2.915m. On003 the common-prefix leg RMSE improves0.215→0.189rad,
and pelvis-centered ankle errors improve0.322/0.333→0.213/0.202m, but path drift
worsens. Final standing position maxima are0.430/1.861/2.915m from target and
joint-error maxima0.244/0.209/0.236rad. All standing/tracking qualifications stay
closed. Existing policy1300 with original VR completes all819 turning source
controls but falls during return at1,242/1,569, with0.061163rad hard-range excess.
It is not the preferred public-motion baseline merely because it trained longer.

The four native23 VR attempts are independently rechecked for matching weights,
physics, histories before changed input, protected reference channels, every
received encoder/root sample, substep continuity, effort clipping, FK metrics
and original acceptance decisions. This establishes a source-intent contribution
to this turning fall, not a guarantee across motions. Next useful controller
work must preserve full source intent and address root/foot tracking and002
limit overshoot; do not resume the failed walking fine-tune or silently promote
this different source convention into the live receiver. Physical estimator,
transport, mode return and the earlier damping incident remain unqualified.

Verification:47 repository recorder/buffer tests PASS (36.36s),9 reference-pair
rejection/label tests PASS (4.97s), Ruff PASS for the new repository recorder and
tests. No test skips. Evidence in
`C:/Users/camer/sonic23_sim_artifacts/internet_pico_20260909_v1/`:

- `original29_buffered_baseline_v1/report.json` and its three all29 traces;
  `original29_public_comparison_v1.json`, SHA256
  `2065b64ee997133ddbf803cefd0311da26456871f7e7b825e5f126c0b4c78dfd` (95 inputs).
- `walk002|walk003|walk008/original29_vr_initial_v1/` and
  `walk003/original29_vr_1300_v1/`; `original29_vr_comparison_v1.json`, SHA256
  `db1312f94d24b9bfa9542ce3f4cf94f77a4dcb4fb22fe0c7c2a53592dd25faeb` (89 inputs).
- `ORIGINAL_VR_ABLATION.md` records both bounded experiments; reusable diagnostic
  and independent audit drivers are colocated. Completed offline turning video:
  `walk003/original29_vr_initial_v1/nominal.measured-simulation.mp4`, SHA256
  `362c3a21fd6bd3faab752a77cf2aa4d50b1d52f805e79a59a23d902a44e74d77`.
  ffprobe confirms1,570 frames,50fps,960x720,31.40s (31.38s integrated motion plus
  the initial-state image). Frames20s and30s are visually checked. This is saved
  actual simulator qpos, not new controller execution or physical-robot footage;
  the tracking camera is not a visualization of source path error. Render receipt
  explicitly retains failed tracking and all unqualified flags. No jobs remain
  running from these seven nominal comparisons or their offline audits.

## 2026-09-08, 18:55 UTC (Sep 9, 04:55 Sydney): complete public walking contact repair passes sampled geometry; bounded walking fine-tune fails CPU tracking

This goal turn is PROGRESS, not deployment qualification. The remaining offline
hand–hip collision is fixed without changing physical limits or collision
tolerances. Full-body controller tracking is still failing. No physical robot,
DDS, mode transition, gain/effort-limit change, policy promotion, commit or push.
The prior physical damping incident is not diagnosed by these simulations.

The contact SQP solver now retries smaller **intermediate** trust boxes only when
a feasible normal step fails its nonlinear line search. Factors0.1 and0.01 keep
the same objective, all relative weights, reference bounds and final acceptance.
Successful normal steps and no-step behavior remain unchanged. Each proposed step
is independently checked for finite shape and trust-box containment, and must
pass the original correction audit and reduce nonlinear merit. This is not a
new physical action limit or a relaxed collision gate.

The complete fresh `contact_adaptive_trust_v5` fit takes21 accepted SQP steps.
The final factor0.01 step reduces merit from6.437e-6 to8.759e-7. Independent
serialized recomputation finds **zero physical self-collision samples on both
667-control and1,333-half-control grids**, minimum sole gap+0.099968 mm, and
joint/rate bounds PASS. All667 imported50-Hz samples, original speed, root
quaternion and waist yaw are preserved. This is sampled geometry, not continuous
collision proof or validation of the raw recording's fractional final endpoint.
Conditional unsmoothed support screening still has649 no-wrench rows and18
within-effort solutions; no dynamic feasibility or standing acquisition claim.

A separately serialized same-point trust witness also passes geometry; it and
the fresh full fit are not bit-identical (maximum mixed pose-channel difference
1.1921e-6). The **witness**, SHA256
`a1a6a3da64eb88234a444610eb34f3fbe9ea09b61b073f7a2aa7f4d56099f5d5`,
is the unchanged source used for the following CPU and training comparisons.
It is explicitly not an accepted frozen BONES training-bank member.

Two complete nominal CPU attempts on that repaired witness:

| Actor | Lifecycle controls | Source controls | Source root-position p95 | Source leg-joint RMSE | Source landmark screen |
| --- | --- | --- | --- | --- | --- |
| Fresh released-weight native23/root-feedback initialization |813/1417; falls|463/667|1.367 m|0.226 rad|FAIL|
| Existing policy1300 |1417/1417; no absolute fall stop|667/667|1.421 m|0.177 rad|FAIL|

These per-run metrics use different available source lengths; the independent
comparison also reports their common463-control prefix. Policy1300's final
standing position is0.444 m from its target. Completion is not successful
tracking or standing qualification. Measured root state remains privileged
simulator state; research absolute stops are not the live receiver's fallback.
The audit rechecks received inputs bit-exactly, frozen encoder tensors/tokens,
initial states, physical model, and delayed scoring frame11+t.

175 focused tests PASS, including unchanged base/no-step behavior, tightened
trust rows only, valid backoff, rejection of invalid steps and no qualification
leakage. Scoped Ruff PASS. Historical pre-backoff helper is preserved under
`pre_contact_trust_backoff/`; old receipts are not relabeled as current runs.

**Controller experiment completed and rejected as a motion solution:** existing
explicit local `regression` mode, fresh initialization,32 environments,100 updates,
64 rollout steps,204,800 transitions, approximately10m44s of logged training, same
v4 objective, source-projection auxiliary term, feedback-priority optimizer,
200-ms received-source semantics and pinned native physics. The actual
checkpoint0 was saved before training updates and then exported; its fresh
repaired002 and raw003 rollouts reproduce prior qpos/qvel/encoder traces
bit-exactly. Training and CPU lifecycle arrays also agree bit-exactly. Execution
receipts pass nominal model-parameter, source/action and buffered-input checks;
this is not a claim of identical CPU/GPU numerical engines.

Independent update verification passes: all18 decoder tensors change, all10
encoder tensors remain exact, all36,864 conditioner weights change, and
critic/optimizer state is finite. Encoder ONNX parity is exact; trained decoder
maximum parity error1.43e-6. Actual CPU motion results still reject the candidate:

| Source | Initial ->100 lifecycle controls | Matched source controls | Root-position p95, initial ->100 | Leg-joint RMSE, initial ->100 |
| --- | --- | --- | --- | --- |
| Repaired public002 |813 ->812 /1417; both fall|462|1.354 ->1.560 m|0.226 ->0.213 rad|
| Raw public003 development regression |1068 ->1096 /1569; both fall|718|1.329 ->1.544 m|0.215 ->0.200 rad|

Every source landmark screen FAILS. Small joint-error reductions do not offset
worse world-space drift or incomplete source/return coverage. Mixed training
resets report86 completed reference suffixes but **zero complete reference
timelines**; neither is successful motion qualification. This short, single-seed
local experiment does not establish that more appropriate native23 training is
impossible. It does reject the claim that this bounded walking fine-tune makes
teleop ready. No automatic continuation of this block or candidate promotion.
Remaining work is a demonstrably better dynamic tracking method/reference,
then complete SIM lifecycle and live transport/state-estimator validation—not
further bookkeeping or relabeling sampled geometry as controller success.

Evidence: `C:/Users/camer/sonic23_sim_artifacts/internet_pico_20260909_v1/`,
`contact_adaptive_trust_geometry_audit_v5.json`,
`contact_trust_initial1300_comparison_v1.json`,
`walk002/contact_adaptive_trust_v5/`,
`walk002/contact_trust_serialized_v1/`, and
`walk002/public_walk_regression_v1/OUTCOME.md` and its unchanged `EXPERIMENT.md`.
Fresh-fit geometry audit SHA256
`d778ff78239f10291c4f717cfbe9e39471da3c0d969a98f59dff82e2a2a35bc1`;
initial/1300 contact-reference comparison
`be03555867895febc1bcfa40c7abb43ba2268a49da252f6be78b858823891ce8`.
New training comparison002 SHA256
`71f5e02a87cfcebb80284c14254945060de104dd0da75bc14111e50434346120`,
comparison003 `c310759bf7891bd8572481846116ab76df72b126367d2cf62318f571469e1d0b`.

## 2026-09-08, 18:04 UTC (Sep 9, 04:04 Sydney): moving-source boundary and numerical scaling fixed; full public walking repair still rejected

This goal turn is PROGRESS, not deployment qualification. Two reproducible
contact-retargeter defects were fixed, and a separate lost-evidence failure was
fixed. **No policy training, policy promotion, physics/gain/limit changes, DDS
or physical robot commands.** Earlier public-motion controller failures remain.
This does not diagnose or fix the physical robot's prior damping incident.

The public `walk002` source begins moving. Its fixed waist-yaw displacement is
-0.068660609 rad in the first20 ms, or -3.433030 rad/s. The contact optimizer
previously assumed zero initial joint velocity while keeping that waist path
immutable. Its tightened first-acceleration allowance is only0.03184 rad:
an unavoidable0.036820609-rad row violation. The new **explicit diagnostic**
initial-velocity option carries the same declared boundary into optimization
and final audits. Default remains stationary. This is not evidence of acquiring
the moving source from normal standing, and does not weaken velocity limits.

A same-serialized-iterate comparison then reproduced `NumericalError` with the
old summed objective, versus an independently feasible `AlmostSolved` result
when the whole objective was divided by667 control frames. Original-row error
was1.12e-10; norm error0. This common positive scalar changes neither relative
weights nor hard constraints. Production contact fitting now uses1/frame count.
`AlmostSolved` never establishes optimality; strict independent feasibility
and nonlinear acceptance still apply.

The complete667-frame,50-Hz walking reference was fitted with all samples and
timing retained. Root attitude and waist yaw remain bit-exact; no joints added.
Independent serialized geometry comparison:

| Diagnostic | Accepted SQP steps | Self-collision samples /1,333 at100 Hz | Worst overlap | Joint/rate bounds |
| --- | --- | --- | --- | --- |
| Zero initial boundary, original objective | 0 | 480 | 50.638 mm | FAIL |
| Explicit source boundary, original objective | 2 | 242 | 12.427 mm | PASS |
| Explicit source boundary, normalized objective (`mean_v4`) | 20 | 1 | 0.208 mm | PASS |

Final remaining contact is left rubber hand against left hip-pitch link at
source9.32 s (control466 /half-grid932). Minimum sole gap is+0.099975 mm.
Maximum joint velocity4.975003 rad/s and acceleration79.600215 rad/s2 remain
inside existing5/80 limits. Last SQP step is feasible but no tested line-search
fraction reduces nonlinear merit; no convergence or global infeasibility claim.
**Geometry remains rejected; collision tolerance was not raised.**

The saved-iterate follow-up independently tests all eight existing line-search
fractions. None is collision-free, and all increase contact merit. Thus this
probe does not find a valid candidate hidden by the merit-decrease gate; changing
that gate is not justified. It reconstructs serialized coordinates, not the
previous internal float64 iterate. A separate read-only probe at control466
finds query/physical signed distances agree within1.42e-8 m; maximum analytic
versus finite-difference query-gradient discrepancy is4.83e-6 m/rad. No large
collision-query sign or margin mismatch is found at this remaining contact.

The first normalized run (`mean_v3`) aborted in the subsequent support LP with
HiGHS `Unknown`, before the old script saved any fitted arrays. Console-only
failure is documented separately, not relabeled as a complete receipt. The CLI
now saves a hash-bound, explicitly incomplete `geometry_only.json` and motion
before support solving. Optional diagnostic recording retains numerical support
errors as non-passing, indeterminate rows. Default support failure still raises.
The fresh `mean_v4` rerun reproduced the same optimizer results and completed:
647 frames have no support-wrench solution under the stated candidate cones,
1 is numerically indeterminate (frame166), and19 have conditional solutions
within effort limits. These unsmoothed exact-reference screens ignore
self-contact assistance and do not prove nearby adapted motion impossible.

169 focused tests PASS, including boundary copying/validation, objective-scale
equivalence, strict bounds, unknown-support retention, incomplete stage flags
and overwrite refusal. Scoped Ruff and tracked diff checks PASS.

Independent audit binds133 inputs, verifies old-source snapshots instead of
relabeling historical runs, independently recomputes all serialized joint/rate,
FK, floor and physical self-contact results, and checks the three raw recordings.
Across all1,107 unresampled raw frames, twelve leg-body positions agree with
original29 FK within1.748 micrometres. Other shared-body offsets differ by up
to10.143 mm, so identical source asset bytes are not claimed. Leg-joint order
is not an explanation for the large tracking error. Scorer code also already
uses delayed next-reference frame11+t, not newest emission19+t; a missing
200-ms scoring shift is not the explanation.

Evidence root: `C:/Users/camer/sonic23_sim_artifacts/internet_pico_20260909_v1/`.
`public_source_boundary_contact_comparison_v4.json` SHA256
`802e3a02ab061dc7fffd1b3335b9d7d35142fefc5b88516838c0a6d018d3e028`;
`walk002/contact_objective_normalization_probe.json` SHA256
`54f4f1511cf05c3ae796293e9a5aea0e86cae90d5c5c2991a75acf7e252ff6c5`;
`walk002/contact_final_line_search_probe_v1.json` SHA256
`3edc0ceac6e63ee2da8d4b5fe15fd4f502d5b66894db546159e1e4ac6d4443b4`;
`walk002/contact_conditioned_source_velocity_mean_v4/`; preserved old code in
`pre_contact_boundary_fix/`, `pre_contact_mean_normalization/` and
`pre_support_failure_retention/`. No new reference is an accepted training-bank
member. Full-body teleop and arbitrary29-to23 dance parity remain unqualified.

## 2026-09-08, 17:11 UTC (Sep 9, 03:11 Sydney): real-reference bounds audited; absolute-tilt retargeting bug fixed; geometric fit is not a controller fix

This goal turn is PROGRESS: full-frame source/training-bank audits, an actual
retargeter defect fixed, three complete constrained reference fits, and three
new matched-policy CPU dynamics runs. **Deployment remains unqualified.** No
robot, DDS, gains/effort-limit changes, new training, policy promotion, commit or
push. The new fits are rejected as a deployment solution: all tracking screens
still fail, and two simulated motions fall.

The actual latest 1200-to-1300 training segment uses only body-check and idle
source phases. Their planar root-speed p95 is 0.041/0.068 m/s; public 002/003/008
is 0.847/1.136/1.503 m/s. This is a statement about that verified curriculum,
not the entire previous training history or original SONIC pretraining.

The untouched imported public references lie inside native hard joint ranges,
but exceed the **unchanged reachable controller target envelope** on
587/667, 368/819 and 212/364 frames. The main mismatch is knee extension:
source requests reach -0.087267 rad; the default safe target lower bound is
+0.073106 rad. Some 003/008 ankle-roll and 008 wrist-roll requests also exceed
reachable targets. These are reference targets, not measured motor violations.
No target, hard limit or fallback threshold was widened.

Source sole geometry is also inconsistent with flat-floor contact: nearest-sole
gaps range approximately -21 to +31 mm (002), -20 to +60 mm (003), and -17 to
+58 mm (008). Existing nominal-support diagnostics find only 6/667, 0/819 and
0/364 conditional solutions. Those unsmoothed finite-difference, 2-mm contact-
candidate LP screens are necessary-condition diagnostics, not proof that a
nearby adapted motion is impossible or the cause of any controller failure.

### Actual code fix and full-reference experiment

`OriginalTaskPath` previously bounded the **relative rotation correction**, not
the resulting absolute pelvis tilt. Public 003 frame571 has 0.453511-rad pelvis
tilt; a 0.1-rad correction increasing its tilt passed the old relative/temporal
audit despite resulting in 0.553511 rad. This is a static three-copy constraint
witness from a real source frame, not a fabricated full-motion success.

The v2 fitter now enforces absolute tilt <=0.5 rad separately: exact cosine
Jacobian rows in SQP, strict original-row reaudit, nonlinear line search, and
serialized-path audit. An explicitly feasible root-quaternion seed is allowed;
an infeasible seed is rejected, never used to justify widening the limit.
202 focused tests PASS, including finite-difference Jacobians, near-limit
fitting/serialization, forged solver output, all original-time/hand/collision
consumers, root-feedback replay and source-initialization tests. Scoped Ruff
and git diff checks PASS.

All 667/819/364 frames were fitted at original 50-Hz timing, without cropping,
slowing or dropping joints. Fits use the original29 task targets and unchanged
native physics, with existing stricter 9.5 action / 0.05-rad guarded reference
bounds, joint speed <=5 rad/s and acceleration <=80 rad/s2. Root correction is
bounded per XYZ axis (+/-0.08 m), not by an 8-cm Euclidean radius. Joint-change
bounds are relative to the disclosed feasible seed, not the raw source.

| Clip | Max original29 foot-target error after fit L/R (mm) | Max reference pelvis tilt (rad) | Trained controller completed/requested after fit |
| --- | --- | --- | --- |
| 002 | 0.157 / 0.195 | 0.237950 | 1417/1417; tracking FAIL |
| 003 | 1.035 / 1.023 | 0.497500 | 1067/1569; fall, versus prior 1130/1569 |
| 008 | 3.879 / 1.754 | 0.327448 | 695/1114; fall, versus prior complete 1114/1114 |

Each controller request retains the complete source plus the same standing,
entry, return and proof durations; fitted endpoints change the generated ramp
targets. Same policy1300, initial state/history, model, gains, codec and received
200-ms runtime. This research referee has no nominal balance fallback and stops
below 0.12 m or above 2.2-rad absolute tilt; it is not the live 0.5-rad fallback.
All failures and prefixes remain recorded. Kinematic bounds and sub-5-mm foot
reference fidelity did **not** produce dynamic tracking. No new training bank
or live retargeter is accepted from this experiment. Whole-source fitting uses
future frames and is not a causal teleop implementation.

Independent `constrained_reference_policy_comparison.json` binds 117 inputs,
reconstructs every geometric seed/path audit from the pinned raw recording,
rechecks serialized original29 task errors, and validates every received policy
input and measured landmark by independent FK. Same initial qpos/qvel/history,
physical parameters and policy identity are verified exactly; both old and new
runs are scored against the **same unchanged native23 source**, not only each
run's different fitted target. Equal prefixes are 667/717/345 source controls.

| Clip | Original-target root p95 before -> fitted input (m) | Original-target leg RMSE before -> fitted input (rad) |
| --- | --- | --- |
| 002 | 0.519638 -> 1.233072 | 0.200356 -> 0.318986 |
| 003 | 1.069628 -> 0.860653 | 0.229514 -> 0.294421 |
| 008 | 1.776980 -> 2.874390 | 0.224907 -> 0.301633 |

003's smaller root-position error does not offset its earlier fall and sharply
worse pelvis-relative foot error. No blanket improvement claim. Comparison
SHA256 `dd73ba70df8fa51699188ce2a4d72ec428446ee2c4b1c3cb5d7806dcdecd9b5d`.

Evidence: `C:/Users/camer/sonic23_sim_artifacts/internet_pico_20260909_v1/`:
`public_training_bank_reference_shape.json` (SHA256
`61b5d1d3748e1435784aa71a828ed175898c2ec8aa69cf52a5fcdc6aea9ab782`),
per-clip `native_nominal_source_support_v1.json`,
`constrained_task_reference_v1/`, and `constrained_task_buffered1300_v1/`.
The pre-fix helper is retained in `pre_absolute_tilt_fix/` with SHA256
`4270177ea9805f44d0fed0f68a56fea1cf966d6ef0cbb474a05e66f626bd43e6`;
historical receipts are not relabeled as runs of the new fitter. Physical robot
damping remains a separate unresolved incident; this SIM finding does not
diagnose it. Next policy work requires contact-consistent broader references
and held-out dynamic evidence, not another promotion of a geometric-only pass.

## 2026-09-08, 16:33 UTC (Sep 9, 02:33 Sydney): newer buffered policy tested on public motions; entry alone does not explain failure

Previous goal turn was PROGRESS: source-action ablation and original29 geometry
loss were measured. This turn is PROGRESS: nine new CPU dynamics runs, explicit
nominal-only entry-isolation tooling, and independent input/FK/encoder audits.
No new training, physical robot, DDS, hardware launcher, gains, limits, commit,
push or policy promotion. Goal remains ACTIVE and deployment remains unqualified.

The newer trained root-feedback actor cannot be substituted into the older live
receiver: it requires the explicit received-source 200-ms q0 horizon and a
separate nine-value current root-state input. Tested its already-matched CPU
runtime before implementing another live integration. Selected the documented
corrected-step policy1300, not the later rejected reward branches, plus its
source-initialized diagnostic and the untouched zero-update buffered actor.

Policy1300 checkpoint SHA256 `e591f475f1cc173de8d7fd1a1f092a740aa5e87278b69bc30cf82cc8745d0f45`,
decoder `83c5c6bb71ea96b3dbe6717cec1e0a39e64ee2bd389d082ae7def7fcfebf3c28`.
Zero-update checkpoint `e72a28668a933b0b836bc81bf6306946b3df52f430b278b08f8b810c29022304`,
decoder `0ab80b0b4e461ff71059d81ebc4679c00c483027dbf32fa9aac258fa4d8817a1`.
Both execute compatibility `308a6d89fedc33724f14fcf41ca0a6775986a2edc4581e47339b3866240c82e2`.
Their encoder file hashes differ, but initializer tensors and encoder outputs
on the union of all 9,885 recorded inputs are exactly equal. Frozen-encoder
identity was checked numerically, not inferred from metadata alone.

### Full requests, retained failures, matched comparisons

Each lifecycle includes 250 standing, 100 acquisition, every source frame,
100 return, 250 returned-standing and 50 proof controls. Ten extra scripted
constant-standing input samples feed the delayed buffer after the scored tail;
they are disclosed generated inputs, not captured PICO samples or EOF draining.
Generated ramps remain unqualified kinematic requests, not contact plans.
Nominal-only cases here are not a three-case perturbation campaign.

| Public clip | Source frames | Initial actor completed/requested | Policy1300 completed/requested | Policy1300 source-initialized completed/requested |
| --- | --- | --- | --- | --- |
| 002 | 667 | 1417/1417 | 1417/1417 | 1067/1067 |
| 003 | 819 | 1068/1569, fall | 1130/1569, fall | 794/1219, fall |
| 008 | 364 | 1114/1114 | 1114/1114 | 764/764 |

All nine source tracking screens FAIL, including completed runs. This research
referee has no balance fallback and its existing absolute stop is height below
0.12 m or tilt above 2.2 rad. It does not reproduce the older live controller's
0.5-rad nominal fallback: longer survival must not be called a deployment gain.
All three 003 cases fall below 0.12 m; measured hard-limit excess reaches
0.005479 / 0.026378 / 0.007810 rad respectively. No thresholds were changed.

Initial versus trained comparisons use identical full references, exact initial
qpos/qvel/history, physical model, gains, action codec and reference timing.
Source metrics below use equal prefixes (667 / 718 / 364 controls), retaining
the initial actor's failed 003 endpoint rather than hiding it:

| Clip | Root-position p95 initial -> trained (m) | Leg RMSE initial -> trained (rad) | Pelvis-relative ankle p95 L/R initial -> trained (m) |
| --- | --- | --- | --- |
| 002 | 0.651309 -> 0.519638 | 0.186297 -> 0.200356 | 0.170870/0.158132 -> 0.180046/0.163581 |
| 003 | 1.328610 -> 1.069614 | 0.214906 -> 0.229674 | 0.321967/0.332722 -> 0.254814/0.271294 |
| 008 | 2.554486 -> 1.771722 | 0.207007 -> 0.233960 | 0.210507/0.144997 -> 0.248113/0.216361 |

Training improves root-path error but worsens leg-joint error on all three;
ankle-relative errors improve only on 003's matched prefix. These ankle-origin
metrics are not sole contact/slip proof or original29 task fidelity. They are
not directly substituted for the previous recorder's differently scoped metrics.

Source initialization omits only 350 initial standing/acquisition controls and
uses eleven explicitly synthetic source-zero prehistory samples. All original
source channels and the entire return tail are byte-for-byte preserved as
arrays; no postinitial pose rewrite. This does not rescue any clip: 003 still
falls after 794/819 source controls. On full 002/008, source-initialized root
p95 is 0.733322/2.167400 m versus normal-entry 0.519638/1.771722 m. Therefore
the earlier two training-motion entry diagnosis is not a sufficient explanation
for these public recordings. Swapping in policy1300 or fixing entry alone is
not a supported deployment solution; broader motion control remains unresolved.

`diagnose_g1_true23_source_initialization` now supports explicit
`--nominal-only-baseline-diagnostic` with a distinct v2 diagnostic kind. Default
three-case requirement remains; duplicates, missing cases, prefixes and relabeled
source-initialized baselines are rejected. Failed full requests remain valid
diagnostic evidence, never successful qualification or continuation parents.
96 focused tests PASS, including real two-input ONNX parity; scoped Ruff PASS.

Evidence at `C:/Users/camer/sonic23_sim_artifacts/internet_pico_20260909_v1/`:
each clip retains `buffered_initial_lifecycle_v1/`, `buffered1300_lifecycle_v1/`
and `buffered1300_source_initialized_v1/`. `buffered_training_entry_comparison.json`
binds 95 inputs and rechecks every received encoder/root feature, every recorded
landmark via independent FK, source preservation and matched initial states.
No additional dynamics run is performed by `audit_buffered_public_comparison.py`.
No current hardware-damping diagnosis follows from these SIM results. Public
clips remain development cases, not untouched final qualification evidence.

## 2026-09-08, 16:08 UTC (Sep 9, 02:08 Sydney): source-action ablation improves joints modestly; public turn still fails, original29 reference loss quantified

Previous goal turn was PROGRESS: original V14 comparison and geometry-only
ablation improved arm tracking but not lower-body fidelity. This turn is also
PROGRESS, not deployment completion: three full saved-public-motion dynamics
runs, an independently checked action-projection audit, and a separate original29
versus native23 reference-geometry audit. No robot commands, DDS, training,
gains, physical gates, model promotion, commit or push.

Added explicit offline-only `--source-action-units-diagnostic` to the recorder.
It requires validated paired reports, uses the existing source action-history
codec, and maps the released source-scale linear target request through the
existing bounded inverse transform. Native target bounds, raw-action guard,
actuator effort and fallback remain unchanged. Reports retain source/inverse
actions and projected-target deltas; the tracker records the action convention.
This is an ablation on existing breadth25 weights, not a claim that their
training used matching source-action semantics. The live/physical launchers
do not receive this diagnostic switch.

All runs retain calibrated source orientation and the preceding virtual-source
reference geometry. Same encoder/decoder bytes, native23 model, packets and
exact initial qpos/qvel. walk002 completes 656/656 SONIC controls and walk008
353/353, both without fallback. walk003 completes 561 SONIC plus 247 fallback
controls; nominal fallback starts at 11.22 s versus 11.12 s previously. It is
still a FAILED full-body replay, not a repaired turn.

Matched SONIC-prefix comparison against geometry-only (656 / 556 / 353 controls):

| Clip | Leg RMSE before -> after (rad) | Foot p95 L/R before -> after (m) | Root path p95 before -> after (m) |
| --- | --- | --- | --- |
| 002 | 0.227015 -> 0.199275 | 0.272466/0.218482 -> 0.264848/0.213915 | 1.173812 -> 1.011741 |
| 003 | 0.245428 -> 0.226839 | 0.375310/0.352269 -> 0.353416/0.351652 | 1.104338 -> 0.742511 |
| 008 | 0.235987 -> 0.209306 | 0.237169/0.298080 -> 0.270859/0.313325 | 3.328864 -> 2.908011 |

Leg-joint RMSE improves 7.6-12.2%, but foot error remains 21-35 cm and worsens
on 008. Source XY is not an explicit command in the 267-input interface;
foot error and fallback remain independent reasons to reject fidelity.
Compiled original C++ target constants independently reproduce recorded target
projection within 2.1e-7 rad. At least one requested target is projected on
464/656, 395/561 and 270/353 accepted SONIC inferences (70.4-76.5%); maximum
projection is 1.052967 / 0.879630 / 0.888616 rad. Applied targets stay inside
the unchanged native inner envelope. This saturation is not permission to
loosen limits, nor does it mean source motion joint samples were clipped.

### Original29 geometry is a separate limitation, not the fallback diagnosis

Audited all 667/819/364 resampled source frames using the pinned public joint
records, local original SONIC29 XML and native23 XML, with zero physics steps.
This does not assert the local original XML is byte-identical to TWIST2's model.
Direct joint selection changes hand position p95 by 0.126/0.143 m (002),
0.239/0.217 m (003), and 0.184/0.231 m (008), before control runs. Foot reference
positions are identical in these models; dynamics/COM differences are not ruled
out, but changed foot reference positions cannot explain the measured errors.

The native pelvis-to-torso chain has only waist yaw: their Z-axis tilt is equal.
On 003 the original29 torso reaches 0.992403 rad tilt, with 249 source frames
above the current 0.5-rad nominal fallback threshold. Exact torso reproduction
inside that envelope therefore has counterexamples. At frame 571 the original
torso tilt is 0.983395 rad while the direct native import is 0.453511 rad.
This establishes a limit of this mapping/envelope, not ultimate hardware
infeasibility and not the cause of the measured controller failure. The latter
still exceeds its already-projected native reference: at fallback, measured
pelvis tilt is 0.507797 rad against 0.453511 rad requested. Raising a threshold
would not qualify control. Physical robot damping remains separately unexplained.

Evidence root `C:/Users/camer/sonic23_sim_artifacts/internet_pico_20260909_v1/`:
`source_action_comparison.json` binds before/after reports, traces and the C++
projection check; each clip's `paired_source_action_v1/` retains report, full
measured trace and tracking. `original29_reference_loss.json` binds source,
model, threshold-report and `audit_public_reference_loss.py` hashes.

93 focused tests PASS. New tests cover action/history conversion, clipping
receipt, invalid/nonfinite source action rejection, and empty traces. Public
clips are development cases, not untouched held-out qualification. Full-body
tracking, ordinary-standing acquisition/return, real-headset and supervised
hardware evidence are still missing. Goal ACTIVE; no candidate promoted.

## 2026-09-08, 15:28 UTC (Sep 9, 01:28 Sydney): original walk policy compared; reference geometry reduces arm error, not whole-body failure

Previous goal turn was PROGRESS: paired localhost SIM integration, exact recorded-
state parity and fault reporting were implemented and tested. This turn compares
the original `23dofsonic` walk policy on the same three new public recordings,
then isolates the already-documented source-reference geometry correction.
No training, robot commands, DDS, hardware pins, gains or physical gates changed.

### Original V14 comparison

The original mode registry selects encoder `733353148bef1eb8dd83a96416b7a89f0b5c3530ceb9e0cec9c25fdb04f56ff2`
and V14-100 decoder `f66408ae9a10720a3aff717269d0e2a4e07ab471e449a6fe8f5bae5e8607ef63`.
This is the original exported walk profile, not the later retrained
`v14_native_ieee` experiment. New explicit `--original-native23-v14-diagnostic`
in the offline recorder binds those bytes; it cannot mix paired report overrides.
The live CLI does not expose this mode. Used the same once-calibrated source
orientation, native23 physics, source initialization, bounds and active fallback
as the current paired breadth25 tests. Original launchers that disable fallback
were not run or changed; historical survival reports are not fidelity evidence.

- walk002: original fails after 188 integrated controls / 3.76 s on the action
  guard; paired breadth25 completes 656/656 without fallback.
- walk003: original reaches fallback after 352 SONIC controls / 7.04 s, then
  fails its physical gate after control 357 / 7.14 s: height 0.440189 m,
  tilt 0.578346 rad. Paired breadth25 reaches 555 SONIC controls before fallback;
  its full 808-control report also remains failed.
- walk008: both finish 353/353 without fallback. Original hand pelvis-relative
  p95 is 0.386704/0.385979 m versus paired 0.241275/0.239762 m; original foot
  p95 is 0.341788/0.357969 m versus paired 0.277125/0.303779 m. Original root
  path p95 is better, 2.605937 m versus 3.269564 m; do not omit this regression.

The first original002 run exposed a recorder counting a pre-integration action
rejection as another physical state. Fixed the recorder to retain the failure
and attempted/completed counts without fabricating another 20-ms sample. Added
explicit `failed_attempt_integrated` and matching tracker validation/tests.
Original first report is preserved in `original_v14_orientation/`, superseded
for tracking by `original_v14_orientation_v2/`; no failure was relabeled a pass.

### Geometry-only ablation on current paired weights

Measured walk002 elbows average 1.295/1.303 rad in the source but only
0.136/0.117 rad in the paired native-FK rollout, despite valid source arm ranges.
September 7 work already isolated a reference-geometry cause; this is not a new
discovery. Reused `virtual_source_vr_terms`: embed current q9 native23 angles in
original29 kinematics with absent axes zero, supply only its 21 VR reference
values to the encoder. New `--virtual-source-reference-diagnostic` in the bounded
offline recorder precomputes independent frames, with a unit test proving each
frame equals its standalone computation. No future preview or policy target
replacement; orientation, reference joints, measured state, reward/metric truth,
physical native23 model, raw transform and fallback are unchanged. This is an
explicit encoder-input convention ablation, not a relabeled training lineage.
Reference-only original29 XML SHA256:
`386b1bb9ea5b69ccd6fd0283a73ffea1ee052df95564e23a780125fbcbe2c645`.

All three geometry runs preserve complete requested durations. walk002 and
walk008 finish without fallback. walk003 falls back at control 556 / 11.12 s,
then integrates 252 fallback controls: still FAIL, not a meaningful balance gain.
Comparison verifies exact equal initial qpos/qvel, identical packet/model hashes
and unchanged encoder/decoder bytes. Errors below use matched SONIC prefixes
(656 / 555 / 353 controls), not unequal failure-truncated distributions:

| Clip | Arm RMSE, native -> virtual (rad) | Hand p95 L/R, native -> virtual (m) | Foot p95 L/R, native -> virtual (m) |
| --- | --- | --- | --- |
| 002 | 0.596487 -> 0.148292 | 0.162410/0.175471 -> 0.109474/0.124708 | 0.266102/0.208334 -> 0.272466/0.218482 |
| 003 | 0.583517 -> 0.142069 | 0.165277/0.191515 -> 0.137629/0.138361 | 0.356247/0.318325 -> 0.375420/0.352357 |
| 008 | 0.520299 -> 0.253064 | 0.241275/0.239762 -> 0.226063/0.190377 | 0.277125/0.303779 -> 0.237169/0.298080 |

This reduces arm error by 51-76%, but feet regress on two clips. Root path p95
changes 1.056420 -> 1.173812 m (002), 1.657791 -> 1.104361 m (003 matched prefix),
and 3.269564 -> 3.328864 m (008). Source XY is not directly commanded by this
267-input interface; foot errors and fallback independently remain unresolved.
No new geometry default in live/hardware launchers. Public clips now inform
development choices and must not later be counted as untouched held-out tests.

Evidence root `C:/Users/camer/sonic23_sim_artifacts/internet_pico_20260909_v1/`:
`baseline_geometry_comparison.json` binds all nine reports/traces and matched
metrics. Each clip's `original_v14_orientation_v2/` and
`paired_virtual_source_v1/` contains complete attempted-run evidence and tracking.
New walk002 geometry video is measured-state playback (657 frames; 13.12 s of
measured controls, 13.14 s video including the initial-state frame),
SHA256 `92c16b49f0df7aa6e8bd19b2c9ba88d2f1e53d75b771c2b087b372bdc57db059`.
It is not a new physics run or hardware evidence.

81 focused tests PASS; scoped Ruff/whitespace checks PASS. Goal ACTIVE.
Controller whole-body fidelity, ordinary-standing acquisition/return, a real
headset session and supervised hardware qualification remain incomplete.
Next control work must address lower-body tracking with a matching reference/
action convention; changing encoders or deleting output joints alone is not
enough. No unqualified candidate promoted; no commit/push performed this turn.

## 2026-09-08, 15:08 UTC (Sep 9, 01:08 Sydney): paired live SIM path verified against recorded dynamics; failure reports retained

Moved the already-verified paired-model and once-calibrated source-orientation
path into the localhost live SIM consumer. The physical launcher is unchanged.
The live CLI now requires paired encoder/decoder reports; historical mismatched
replay requires explicit `--legacy-unpaired-diagnostic` and candidate summary.
Mixed modes are rejected before opening the subscriber. Same native23 model,
gains, limits, fallback and source timing; no new weights or hardware activity.

Fixed a concrete diagnostic loss: controller/fallback RuntimeError now produces
a FAILED JSON report with the terminal state and successful versus integrated
transition counts. Optional NPZ retains the initial state and post-received-
control attempts, including an integrated failing state. Transport-fault hold
states are outside this trace's explicitly declared scope; terminal state is
reported separately. Nonfinite terminal components serialize as JSON null.
Packets are checked for freshness again at actual use, including startup.
No automatic retry after physical failure. New report kind/schema is v2; do not
relabel it as v1 to satisfy a historical readiness auditor.

Actual 50-Hz localhost replays, source values unchanged, timestamps rebased only:

- Paired walk002: PASS 656/656, no fallback, maximum packet age at use 26.465 ms.
- Paired walk003: FAIL; 555 SONIC controls then 253 latched-fallback controls.
  All 808 integrate; maximum packet age 25.506 ms. This is not a teleop pass.
- Explicit legacy walk003: FAIL at 11.90 s; 594 successful step returns,
  595 integrated transitions. Report now retains the original physical failure:
  `fallback physical gate failed: height=0.436192, tilt=0.598368`.
  No retry or failure-state removal; maximum packet age 24.602 ms.
- Paired walk002 disconnect test: publisher stops after 120 packets; expected
  timeout triggers 100 fallback controls. Fault screen PASS, fallback minimum
  height 0.772700 m and maximum tilt 0.142147 rad. This qualifies only that
  simulated transport-fault case, not hardware recovery or standing acquisition.

For all three full source replay cases above, every saved qpos, qvel and
simulation_time value is exactly array-equal to its prior offline diagnostic.
Trace SHA256 respectively:
`00f1695879d36229c3f02e623cfbb5c57e70ef360d55942166e9e39ba90c7a0a`,
`c91c12975a95f14b31ccb1d3bc6ab6f0415cd727c781aec703c1a34d61815c2c`,
`546bee0dcbd07c007175edd6e4706c0dca03dfead99baf4d94e7dc2c0ff385b7`.
Thus network scheduling does not explain these reproduced tracking/fallback
failures. This turn fixes the live SIM test path, not the controller's remaining
20.8-35.6 cm foot tracking errors or the turn failure documented below.

Evidence: `C:/Users/camer/sonic23_sim_artifacts/internet_pico_20260909_v1/paired_live_v2/`
with separate `walk002`, `walk003`, `legacy_walk003` and `timeout` directories.
Each has consumer/publisher JSON and measured trace. A post-run metadata-only
edit replaces the overbroad trace label `includes_failed_integrated_state` with
post-received-control state count and explicit separate terminal-state flag;
saved reports remain unmodified. Trace scope and dynamics did not change.

70 focused tests PASS, scoped Ruff and diff whitespace checks PASS. Added tests
cover physical failure, nonfinite state, latched fallback, failed transport
fallback, startup-aged packets and implicit legacy/mixed-pair rejection.
Runbook updated. Goal ACTIVE. No training, DDS, robot commands, commit or push.
Full-body fidelity, ordinary-standing acquisition and hardware readiness remain
unqualified; passing stream delivery is not deployment readiness.

## 2026-09-08, 14:45 UTC (Sep 9, 00:45 Sydney): public recordings expose fidelity gap; paired SIM path improves heading

Follow-up to the three public TWIST2 tests below. Measured actual dynamics
against the unchanged native23 import, not only survival. Critically, the
September 5 entry already documented the legacy encoder mismatch. The older
live launcher used by the 14:00/14:22 tests STILL pins that mismatched pair.
Those results remain historical transport/stability screens, not correctly
paired policy qualification. No live/hardware pins changed or deployment claim.

Added `measure_g1_true23_saved_teleop_tracking.py`: source/motion/packet/trace/
model hashes checked; packet arrays must reproduce exactly from source motion;
one initial XY/yaw alignment, no Z adjustment, lag fit, per-frame world alignment
or dropped failure states. Initial robot state corresponds to source q10;
first integrated 20-ms state compares to q11, then follows elapsed simulation
time. SONIC-only coverage is separate from all recorded/fallback states.
Reports compare native23 imported FK, not original29 task-space fidelity.

Legacy no-fallback walk002 nevertheless has heading p95 124.974 degrees,
left-hand pelvis-relative p95 0.468518 m, root-position p95 0.997943 m.
Legacy walk008 has heading p95 143.550 degrees and left-hand p95 0.691890 m.
Thus the earlier stability passes do not establish usable full-body teleop.

Updated the offline recorder to REQUIRE a validated encoder/decoder pair by
default. Historical profile replay now requires explicit
`--legacy-unpaired-diagnostic`; candidate-summary and paired-report modes are
mutually exclusive. Paired mode retains all existing controller gains, effort
limits, physical checks and fallback. No runtime relabel/promotion.

Tested all three new clips using existing paired breadth25 encoder
`3806b2b63ebadf4d6cbf9f79b7072f2bf27ab8eb8bc6a9b3042f97739cc5428a`
and decoder `f4416889023eb629656fa189649d8cd071cdc3ae61fc1bfd888d07815d21bdc8`,
plus the SIM-only once-calibrated source-orientation path. These are NOT the
legacy residual decoder or the newer root-feedback research model. Both policy
pair and orientation handling change, so this is not an encoder-only ablation.
No new weights, source repair, speed/gain/gate changes or hardware activity.

- walk002: 656/656 SONIC, no fallback; minimum height 0.737324 m,
  max tilt 0.265008 rad. Heading p95 14.002 degrees; left/right hand
  pelvis-relative p95 0.162410 / 0.175471 m.
- walk003: 555/808 SONIC, then tilt fallback at 11.10 s for 253 controls.
  All 808 integrate without exception, minimum height 0.714835 m,
  max tilt 0.530424 rad, but replay remains FAILED. SONIC-prefix heading
  p95 15.055 degrees and hand p95 0.165277 / 0.191515 m. Its shorter
  SONIC prefix must not be compared as equal-duration to the legacy prefix.
- walk008: 353/353 SONIC, no fallback; minimum height 0.707684 m,
  max tilt 0.271107 rad. Heading p95 14.996 degrees and hand p95
  0.241275 / 0.239762 m.

Heading/task errors improve, but not every measure: paired walk002 arm RMSE
worsens 0.543413 -> 0.596487 rad, root-position p95 0.997943 -> 1.056420 m.
Paired walk008 root-position p95 remains 3.269564 m. Feet relative to pelvis
still have 0.2083-0.3562 m p95 across paired SONIC segments. Global source XY
is not directly commanded by the 267-input interface, so root-path error alone
is not sufficient to reject teleop; foot/task error and fallback also remain.
Current training preserves reference pelvis orientation and source root
orientation/velocity at reset; live/SIM adapter had discarded orientation and
initializer still uses upright/zero horizontal velocity. Standing acquisition
and full-body controller fidelity remain unresolved.

Reports/traces in `C:/Users/camer/sonic23_sim_artifacts/internet_pico_20260909_v1/`:
each clip has `recorded/tracking.json` (legacy) and
`paired_orientation/{report.json,measured_trace.npz,tracking.json}`. Existing
orientation-only failed legacy probe also measured; no result overwritten.
Detailed comparison and source links in `RESULTS.md`. 60 focused tests PASS,
scoped Ruff PASS. Goal ACTIVE. No physical robot, DDS, commit, push or training.

## 2026-09-08, 14:22 UTC (Sep 9, 00:22 Sydney): three NEW public recordings tested; two pass, turn exposes failure

User requested internet/real-PICO motion to test without the headset. Located
the official TWIST2 example recordings, verified current GitHub commit
`d5c7108e9ef82d1b8770e5b692f27a1294f3aa8a`, downloaded walks 002, 003 and 008
into `C:/Users/camer/sonic23_sim_artifacts/internet_pico_20260909_v1/` and
verified each published Git blob. These are retargeted robot recordings from
the PICO-based TWIST2 project, not raw XR24/HMD/IMU packets; per-file sensor
provenance is not supplied. Do not label this a real-headset session.

Added a bounded, pinned importer with a restricted NumPy-only unpickler. It
preserves elapsed time, resamples to 50 Hz, selects established native23 joints
and recomputes native23 FK. No clipping, slowdown, appended holds, policy
training, task-space repair or changed gates. Exact source/derived hashes and
11 packet warmup/tail frames plus final sub-20-ms fraction are documented.

Streaming tests ran one at a time with the older frozen-LoRA decoder
`44d1fb2701f1e65460f1c2c23f676bce4f1d4a44b3b112798dc5034af37946b8`,
native23 physics and fallback enabled. walk002: PASS 656/656, no fallback,
max age 29.618 ms, minimum height 0.729576 m, max tilt 0.281358 rad.
walk008: PASS 353/353, no fallback, max age 19.616 ms, minimum height
0.673905 m, max tilt 0.269256 rad. Short walk008 is not >=500-control sustained
transport proof. These passes measure transport/stability, not full-body fidelity.

walk003 FAILS: publisher emits all 808 packets, but consumer exits on
`fallback physical gate failed: height=0.436192, tilt=0.598368`. Existing CLI
does not write a result on this RuntimeError; failure was not discarded.
Added a separate offline recorder that retains the failing integrated state
without retrying or changing guards. It reproduces the identical height/tilt:
tilt fallback at control 593; failure after control 595, 11.90 s, source frame
604, with 594 successful returns. walk002's recorded replay matches streaming
height/tilt exactly. Measured-state videos for both are saved and sampled frames
visually inspected; original physical plant, not reference animation.

New concrete fidelity finding: `retarget_pico_reference_packet` replaces
`reference_anchor_quaternion_xyzw` with identity every frame after native FK.
walk003 turns strongly but the policy loses that source orientation. One
separate SIM comparison retains orientation with a once-only initial yaw
alignment. It avoids the original height failure (minimum 0.721436 m, all
808 controls integrated), but needs base-tilt fallback EARLIER, at control 349,
6.98 s, maximum tilt 0.516499 rad. Still a FAILED SONIC tracking test, not a
qualified fix. Live consumer, physical runtime and policy weights unchanged.

10 importer/recorder/bundle regressions PASS, including malicious pickle-global
rejection, changed source rejection, preserved timing/joints/quaternion order,
failed-state retention and once-only heading alignment. Scoped Ruff PASS.
Detailed source links, results and videos:
`C:/Users/camer/sonic23_sim_artifacts/internet_pico_20260909_v1/RESULTS.md`.
Goal ACTIVE. No hardware, DDS, mode/arming, new training, commit or push.

## 2026-09-08, 14:00 UTC (Sep 9, 00:00 Sydney): full-body streaming SIM path rechecked; live input absent, deployment not ready

User explicitly retained FULL-BODY teleoperation and requested urgency tonight.
Stopped launching training/trajectory sweeps; prioritized the existing pinned
causal full-body policy through real localhost streaming. Goal remains ACTIVE.
No robot/DDS/motor, mode, arming or recovery operations. No commit or push.

Fresh saved authentic PICO tests used producer and consumer in the same WSL
clock domain, 50 Hz, fallback enabled, original native23 physical XML and the
older frozen-LoRA decoder SHA256
`44d1fb2701f1e65460f1c2c23f676bce4f1d4a44b3b112798dc5034af37946b8`.
This does NOT test or qualify the newer nine-input root-feedback research path.

- `walk001`: 684/684 controls, no fallback, maximum age 35.993574 ms,
  minimum base height 0.723594 m, maximum tilt 0.312957 rad.
- `walk010`: 499/499 controls, no fallback, maximum age 9.171461 ms,
  minimum base height 0.701787 m, maximum tilt 0.423733 rad. The report's
  >=500-control sustained-transport flag remains false; no frame was added.
- Timeout, stale and isolated missing-frame tests each received 120 controls,
  latched the expected fault and held simulated balance for 100 controls.
  Fallback minimum height 0.743602 m, maximum tilt 0.191386 rad.
- The first missing-frame test was run concurrently with two other consumers.
  It hit a 113.296522 ms stale-input fault after 3 controls, before the intended
  gap. Fallback was stable, but that test FAILED and remains preserved. The
  isolated gap retry passed. Loaded real-time execution is not qualified.

These are saved-input transport/stability screens, not real-headset sessions,
unchanged-original29 tracking fidelity, universal motion support or physical
qualification. The simulator resets to the first source pose; ordinary-standing
acquisition and return are NOT established by these tests. Root-feedback learned
policy failures and hardware ownership/handoff limitations remain unresolved.

All fresh reports are under
`C:/Users/camer/sonic23_sim_artifacts/teleop_readiness_20260908_2345/`.
Read-only PICO probes at the start and near midnight both found no connected
headset client, zero trackers, no calibration, no body roles and no tracking.
Service and XRT binding hashes match their pins. `adb devices -l` also showed
no device. Windows Wi-Fi is currently `192.168.1.182`. Requested that the user
connect XRoboToolkit and calibrated full-body trackers; no response received
yet. The next genuine live-input SIM test requires that external connection.
Connecting a headset alone would not make physical deployment ready.

Updated the live SIM runbook: consumer and producer now use the same WSL clock
and known runtimes; removed the guessed hard-coded installed APK hash in favor
of an explicitly verified value. The producer's SOMA-venv CLI imports and help
run successfully; raw XRT capture remains CPython 3.10. Headless live-input
test is first; viewer startup/live producer latency are not yet qualified.

Previously running bounded V17 jobs finished with no new learned policy:
BODY aborted while differentiating a near-contact pair that disappeared; no
BODY candidate accepted. IDLE accepted alpha 1/32, cost
22615.894891476884 -> 21351.04044239425, but source feet 78.600/74.005 mm
and left hand 104.536 mm still fail the existing screen. Independent IDLE
verification was attempted after the numerical fix below and rejected the
old reproduction's changed source binding for `g1_true23_pd_shooting.py`.
It is NOT independently verified or qualified. The verifier was not bypassed.
V17 source archive preserves one optimizer report and 109 Python bindings in
`pd_shooting_20260908_v17_evidence/source_archive.json` before the source fix.

Two new zero-step predictive-rollout tests initially FAILED. Native quaternion
self-differentiation returned a spurious 2.17e-19 rotation for an exactly equal
pose, producing tiny target drift through state feedback. Fixed exact pose
identity only; real arbitrarily small joint deviations and velocity differences
remain visible, with no relaxed tolerance. Added direct regression plus real
grounded/airborne QP/codec/substep rollouts requiring bit-identical states and
targets at alpha zero. All 240 scoped regressions PASS; scoped Ruff PASS.
This numerical issue is not evidence of the earlier physical damping cause.

## 2026-09-08, 13:36 UTC: both V16 refinements verified; placement correction tested and in bounded SIM trials

SIM only. Goal ACTIVE and full scope unchanged. No robot/DDS/motor or mode
operations, policy-training continuation, commit or push. Offline trajectories
remain explicitly ineligible as qualified teachers or live controllers.

Fresh IDLE accepts alpha 1/8 then 1/128, completing all 2029 controls and 741
source controls. Seven larger second-iteration trials fail the original physical
guards; none of those candidates is adopted. Original cost
28763.19161619888 -> 22912.98247466053 -> 22615.894891476884. Independent replay
matches all 13 physical/integration arrays, exact cost, source/contact ceilings
and all four terminal metrics against the fixed initial parent. Final source
feet p95 79.331/74.782 mm; hands 104.739/57.135 mm; head 62.638 mm. Both feet and
left hand still FAIL. Final standing joint/root-position/root-speed/root-angle:
0.097599 rad / 39.320 mm / 0.066702 m/s / 0.022791 rad. All four improve versus
the initial parent; angle is slightly higher than iteration 1, not monotonically
improving at every accepted step. Candidate SHA256
`fb1368eb9f4aa1a1f610e356d038900c0ba348113f2c3f017e8259020e6a1e96`.

IDLE has no non-foot ground penetration. Existing self-contact remains up to
0.911 mm, 290 loaded substeps; foot force peaks 382.20/594.43 N; maximum loaded
penetration 3.004 mm. No contact/slip/hardware qualification follows. Evidence:
`pd_shooting_20260908_v16/idle/{fresh_after_verified,fresh_verification}/report.json`
under `C:/Users/camer/sonic23_sim_artifacts/`.

Entry-loading diagnostics reproduce all 619 prefix states exactly. BODY now
reaches ankle Z 63.632/67.859 mm at the original 80.167 mm requested apices,
with zero minimum load and 5/19 requested-air controls below diagnostic 5 N per
foot. COM XY p95 improves to 26.929 mm, but entry-end foot X errors remain
45.640/22.340 mm. IDLE has only 2/19 left and 0/19 right controls below 5 N;
entry-end XY errors [56.111,47.498]/[40.105,50.248] mm persist. This identifies
placement and incomplete unloading, not a qualified stepping transition.

Added opt-in `com_lift_placement` local-search guidance for both primary and
cached solvers. Adds native ankle world-XY residuals only during generated
entry/return, at weight 25000; keeps existing COM/lift and standing guidance.
Targets come from the unchanged original reference, never shifted to measured
feet. Default zero placement weight preserves previous terms. Original source
and timing, full acceptance objective, physical model/gains/limits, self-contact
ceiling and return tests stay unchanged. Finite-difference gradients, positive
semidefinite Hessian, exact translation response, inactive-source/standing
nodes, cache/reference immutability, zero-weight behavior and CLI protections
PASS. 237 combined regressions PASS; scoped Ruff PASS.

One fresh-current iteration per clip is running under
`pd_shooting_20260908_v17/{body,idle}/fresh_placement_guided/`. Both start only
from their independently verified final V16 trajectories. No V17 improvement
claim yet. Before source changes, V16 archives captured six reports and 112
source bindings in `pd_shooting_20260908_v16_evidence/source_archive_complete.json`;
the earlier four-report manifest remains unchanged.

## 2026-09-08, 13:24 UTC: fresh V16 BODY independently improves; remaining error is entry foot placement

SIM only. Full dance/full-body teleop goal remains unchanged; the user's question
about a narrower first teleop milestone has not authorized replacing that goal.
Previous conversational turn did not modify a policy. Both existing optimizer
handles were revalidated live before continuation; BODY has now completed and
IDLE is finishing its second bounded iteration. No new training or hardware,
DDS/motor, mode, commit or push operations.

BODY accepts both fresh iterations (alpha 1/4, then 1/2): original cost
6413.883733302319 -> 4624.119693791849 -> 2515.2208906591713. Independent replay
matches all 13 physical/integration arrays across all 2461 controls, exact full
cost, original source/contact ceilings, and all four terminal nonregression
metrics. Source feet p95 54.600/22.229 mm; hands 28.075/36.251 mm; head 19.765 mm.
Left foot still FAILS 50 mm. Final standing joint/root-position/root-speed/
root-angle: 0.041843 rad / 14.748 mm / 0.013767 m/s / 0.012421 rad.

No non-foot ground penetration; existing hand/hand penetration 0.453 mm, 60
loaded self-contact substeps, peak foot forces 510.48/727.93 N and maximum loaded
penetration 8.034 mm remain unqualified. Maximum velocity/effort ratios are
0.5435/0.7380; no joint-limit excess. This is an offline future-informed control
trajectory, NOT an improved learned policy or a causal teleoperation controller.
Evidence: `C:/Users/camer/sonic23_sim_artifacts/pd_shooting_20260908_v16/body/`
`{fresh_standing_guided,fresh_verification}/report.json`. Candidate SHA256
`094b3d85b6ef53635e8f152cb53a6538ed0c600d07a8319b2f6c1f8b32eeb61f`.

Read-only native FK decomposition finds left-foot mean source error XYZ
[51.013, 10.709, -1.109] mm; X standard deviation 2.250 mm. Source begins with
45.842 mm forward error. Most residual is persistent foot placement, not a new
large oscillation during the clip. These pose measurements do not establish
contact timing/slip or original29 fidelity.

The bounded derivative-only slow-transition hypothesis is rejected. Recomputing
native inverse dynamics on the same 371 poses at twice the interval increases
no-support counts: BODY entry/return 41/41 -> 83/81; IDLE 47/42 -> 74/83.
Original-interval forces reproduce prior reports within 1e-10. This probe did
not integrate physics or check newly resampled intermediate poses; no slowed
reference, policy training or qualification follows from it.

## 2026-09-08 Sydney, 12:59 UTC: V16 IDLE return correction independently verified; fresh refinements running

SIM only. Learned swing-load policy1400 remains REJECTED (six tracking failures);
no additional training, hardware/DDS/motor or mode commands, commit or push.

Standing-orientation guidance resolves the specific V15 terminal regression.
V16 IDLE accepts alpha 1/16: all 2029 controls and 741 source controls complete;
original full cost 32020.769944820404 -> 28763.19161619888. All five source
landmarks improve over V7. Feet p95 83.405/78.585 mm; hands 105.770/62.472 mm;
head 68.615 mm. Both feet and left hand still FAIL their original 50/100 mm
screens. Final standing joint/root-position/root-speed/root-angle improve to
0.110552 rad / 46.104 mm / 0.075713 m/s / 0.025314 rad. These are relative
improvements, not absolute standing or teleoperation qualification.

Independent replay verifies all 13 physical/integration arrays, exact full cost,
original source and self-contact ceilings, and all four return nonregression
metrics. No non-foot ground penetration; existing self-contact remains up to
0.925 mm, 445 loaded self-contact substeps, peak foot forces 361.48/608.16 N,
loaded penetration maximum 2.675 mm. Candidate trace SHA256
`976e2a2862ea1779b6596db739e6608f222c568ab8c3cb2656c98dd432c0e51b`.
Evidence: `C:/Users/camer/sonic23_sim_artifacts/pd_shooting_20260908_v16/`
`idle/{standing_pose_guided,verification}/report.json`.

223 combined PD/codec/physical-bound/campaign regressions PASS; scoped Ruff PASS.
Shortest-arc orientation derivatives match independent native-pose perturbations;
quaternion sign invariance, terminal-node inclusion, source/cache immutability,
unchanged zero-orientation behavior and fail-closed CLI options tested.

Fresh original-physics optimization is running for BODY from verified V14 and
IDLE from verified V16, bounded at two iterations each, same original source,
timing, acceptance cost, physical model and contact/return guards. Original
linearization caches remain separate from guide quadratics. Outputs under
`pd_shooting_20260908_v16/body/fresh_standing_guided/` and
`pd_shooting_20260908_v16/idle/fresh_after_verified/`. No fresh-result claim yet.

Encoder inspection rules out literally missing leg references: frozen 267-value
input includes ten frames of 12 leg positions and forward-difference velocities
(240 values), plus VR21 and relative orientation6. This does not prove sufficient
latent information or trained dynamic response. Re-expressing saved reference
inverse-dynamics root wrenches as ground pressure shows candidate-support polygon
violations around swing/landing (IDLE about 17 mm, beyond separate floating-reset
artifacts). This is a reference-only diagnosis, not a measured contact proof.

## 2026-09-08 Sydney, 12:45 UTC: actual swing-load policy fails all six tracking cases; branch rejected

SIM only. Goal ACTIVE, NOT dance/live-teleop ready. No robot/DDS/motor or mode
commands, no commit/push, and no promotion of this failed policy or offline targets.

The bounded policy1300 -> 1400 swing-load branch completed 100 real PPO updates
(819,200 new transitions). Continuation audit verifies exact initial actor,
critic, optimizer and counters; all ten SONIC encoder tensors remain unchanged;
all 18 decoder tensors changed and optimizer/critic state remains finite.
Export passes: encoder ONNX parity error zero, decoder maximum absolute error
1.1920928955078125e-6. Trained checkpoint SHA256
`182ca20444f6c10e13d0d9ce30d37e7fb4826d0e8f2681cadcfe4f9287bdaade`.

Both complete native-physics three-case campaigns finished (BODY 2461 controls,
IDLE 2029 controls per case; nominal, standing_push_x, standing_push_y). All six
FAIL the source landmark screen. Independent paired comparison rechecks hashes,
identical complete reference, physics, initial state and thresholds, and recomputes
metrics from saved physical state. No case improves all five landmarks.
Nominal BODY feet p95 71.593/50.937 -> 71.815/70.374 mm; hands
73.579/83.826 -> 95.194/103.288 mm; head 45.855 -> 77.487 mm.
Nominal IDLE feet 97.194/88.273 -> 105.718/181.432 mm; hands
110.725/69.072 -> 139.318/117.896 mm; head 86.384 -> 99.512 mm.
BODY return speed improves 0.015789 -> 0.012452 m/s but root position and
orientation worsen; all four nominal IDLE return metrics worsen. The branch is
REJECTED as a continuation parent; previous evaluated policy1300 remains unchanged.
Evidence: `C:/Users/camer/sonic23_sim_artifacts/contact_step_reference_bank_20260908_v1/`
`campaign/segment_1300_1400_swing_load_v1/{body,idle}_matched_comparison.json`.

Reference-force review exposes a separate unresolved issue already recorded in
the V4 reference: BODY entry has 41/371 frames without a conditional support
solution, IDLE 47/371 plus one above effort limits. Each includes 23 initial
frames with no candidate floor contact; additional failures occur around swing
takeoff/landing. This is an optimistic candidate-contact/inverse-dynamics screen,
not measured contacts or proof that all nearby tracking motions are impossible.
Do not treat these references as dynamically validated teachers or continue
reward-only training without resolving the mechanism.

V15 offline IDLE standing-velocity guidance rejects all eight trials. Four
complete candidates reduce original cost and remove the V14 return-speed
regression, but every one worsens original terminal orientation. Closest:
cost 30321.09363, angle 0.02678049175 > fixed V7 0.02676993381 rad.
No tolerance widened; accepted IDLE remains V7. V15 source archive binds one
report and 110 source files before subsequent code changes.

Implemented optional original-standing shortest-arc orientation guidance in
addition to velocity guidance, available in both fresh and cached offline
solvers. Uses the original standing quaternion and native tangent Jacobian;
positive-semidefinite Gauss-Newton terms affect local search only. Original
full cost, source/contact and four return acceptance metrics remain unchanged.
Derivative/immutability/CLI checks and bounded V16 pilots are next; no V16
improvement, learned-policy fix or deployment readiness claimed.

## 2026-09-08 Sydney, 12:12 UTC: fresh BODY independently improves; measured swing-load policy training active

SIM only. Goal ACTIVE, NOT dance/live-teleop ready. No robot/DDS/motors, mode
changes, training-parent promotion of offline trajectories, commit or push.

V14 fresh-current BODY accepts alpha 1/8: original cost 7721.13592 -> 6413.88373.
All 2461 controls and 1173 source frames complete. Independent replay matches
all 13 physical/integration arrays and exact full cost; original source/contact
and all four return metrics improve or remain within their fixed V12 ceilings.
Source ankles p95 60.382/34.539 mm, hands 52.534/62.213 mm, head 31.317 mm.
Left foot still FAILS 50 mm. Final standing joint/root-position/root-speed/
root-angle: 0.087987 rad / 36.637 mm / 0.027690 m/s / 0.033699 rad. Maximum
velocity/effort ratios 0.4656/0.9147; no joint excess or non-foot ground contact.
Existing hand/hand penetration 0.455 mm, 332 loaded self-contact substeps and
peak foot forces 741.77/584.98 N remain unqualified; force/slip/robustness and
original29 fidelity are not proven. Optimizer source archive binds two reports
and 108 sources under `pd_shooting_20260908_v14_evidence/`.

V14 entry-loading diagnostic exactly reproduces all 619 initial/entry states.
Left requested apex 80.167 mm, measured 43.260 mm (V12 40.386 mm). Requested-
flight mean load 50.86 N (V12 61.48), minimum 2.26 N (V12 40.51), and 2/19
controls below diagnostic 5 N (previously none). Right apex 51.642 mm, minimum
load 5.32 N, still 0/19 below 5 N. COM XY p95 62.251 mm. Entry-end left X error
52.702 mm persists. Partial unloading is progress, not a successful step.

Fresh-current IDLE accepts none of eight V14 trials. Four complete trials lower
original cost and satisfy source nonregression, but worsen final standing root
speed (and some root-angle values). Best complete trial cost 28766.42826 has
speed 0.0808034 > fixed V7 0.0802673 m/s. Gates unchanged; output remains V7.
Next local-search correction targets this measured return-velocity tradeoff.

Implemented explicit `root_and_upper_feet_swing_load_v6` training objective:
inherits unchanged V4 rewards and adds only actual ground Fz squared over all
ten original 2ms substeps, gated by received-reference minimum native sole
clearance (10-30 mm ramp). Separate named left/right sensors bind original
`floor` geom, not stock `terrain` body. Cost normalization 100 N, weight -2;
stance load, action targets and unreceived future frames are not substituted
for actual flight. This is a reward hypothesis, not contact qualification.
CPU sensor forces match independent mj_contactForce sums; adding sensors leaves
qpos/qvel/warmstart bit-exact for 100 real substeps. Invalid history/clock/floor
configuration fails closed. 99 focused training/reward tests, 49 campaign/reward
tests and two new physical sensor checks PASS; 205 PD regressions PASS; lint PASS.

One bounded policy1300 -> 1400 swing-load branch is RUNNING under
`contact_step_reference_bank_20260908_v1/campaign/segment_1300_1400_swing_load_v1/`.
Same evaluated policy1300, two references, frozen encoder, actor architecture,
original physics, source codec, existing rewards and mixed-start schedule;
only declared contact reward/sensors added. Actual GPU environment has 128
worlds, 23 actions, 2ms/20ms clocks and unchanged received-source inputs. Initial
model1300 saved; optimizer updates have begun. Stops at 100 updates, then exact
continuation/export checks and all six complete CPU cases. No policy1400 result
or adoption claimed. Failed phase-balanced/sole-world1400 branches are not parents.

## 2026-09-08 Sydney, 11:48 UTC: cached trials exhausted; fresh-current dynamics pilots running

SIM only. Goal ACTIVE; dance/live teleop NOT ready. No hardware/DDS/motor or
mode commands, no learned-policy update, training-parent promotion, commit or
push. Existing best BODY remains independently verified V12; IDLE remains V7.

V13 removed only the EXTRA 1 mm planning clearance, leaving actual contact,
joint/speed, source and return gates unchanged. Plain and COM/lift-guided IDLE
each rejected all eight trials. Both reached complete 2029-control trajectories
at alpha 1/32, but original costs 32161.71250 / 32110.28968 exceed fixed V7
32020.76994 and left-hand error regresses; guided return position also regresses.
Other trials fail physical/planning checks or regress further. Complete replay
is not improved tracking. V13 output traces are unchanged V7, not new parents.
Two reports and 109 source bindings archived under `pd_shooting_20260908_v13_evidence/`.

The previous cached comparisons reused V7 iteration-1 dynamics. Primary
optimizer now rebuilds A/B and original cost derivatives at each current
accepted trajectory, supports the actual-state ten-substep contact predictor
and original-reference entry/return COM/lift guidance, and enforces all four
terminal nonregression metrics in addition to original source/physical checks.
Guidance changes only local search; original full acceptance cost, references,
timing, compiled physics and guards remain fixed. CLI rejects unused/invalid
planning options. Targeted regressions: 38 PASS; scoped Ruff passes.

V14 BODY starts from verified V12 and IDLE from verified V7, never a rejected
V13 candidate. Both independently reproduced all 13 initial arrays and exact
original objectives before starting fresh linearization. One iteration each,
regularization 10, COM/lift guidance, zero EXTRA planning clearance. Runs are
active under `C:/Users/camer/sonic23_sim_artifacts/pd_shooting_20260908_v14/`.
No V14 acceptance, foot-unloading solution, policy improvement or deployment
claim yet. Training reward/transition audit is read-only while sources are bound.

## 2026-09-08 Sydney, 11:19 UTC: guided BODY improves and independently verifies; foot unloading and IDLE remain unsolved

SIM only; SONIC policy unchanged, no hardware/DDS/motors, no training-parent
promotion, commit or push. Goal ACTIVE. Dance/live teleop is NOT ready.

V12 entry/return COM + under-lift guidance accepts a larger full BODY step
(regularization 10, alpha 1/8), using the same V7 cached physical dynamics and
original source/timing/acceptance objective. All 2461 controls and 1173 source
controls complete. Original protected cost 9060.81547 -> 7721.13592. Independent
fresh replay matches all 13 physical/integration arrays, complete objective,
source/self-contact and four terminal nonregression checks against fixed V7.

Source ankles p95 61.851 / 38.489 mm; hands 58.145 / 66.846 mm; head 34.126 mm.
Left foot still FAILS 50 mm. Final standing joint/root-position/root-speed/
root-angle: 0.099950 rad / 41.207 mm / 0.029051 m/s / 0.040199 rad. Final speed
passes fixed V7 nonregression but is slightly worse than V11 0.028966 m/s;
do not claim every metric improves over every prior candidate. Existing hand/
hand penetration 0.457 mm, no new penetrating pairs or non-foot ground contact.
419 loaded self-contact substeps and peak foot forces 759.03/624.19 N remain
unqualified. Neither robustness nor original29 fidelity is demonstrated.

Entry-loading recheck reproduces all 619 initial/entry integration states
exactly. Guidance helps but does NOT create proper foot unloading: first ankle
apex rises 38.278 -> 40.386 mm against unchanged 80.167 mm target. Left-foot
mean requested-flight load 71.93 -> 61.48 N; minimum 59.04 -> 40.51 N. Right
minimum 25.24 -> 11.33 N. Still zero requested-flight controls below diagnostic
5 N. Entry-end left X error 54.47 -> 53.99 mm. COM XY p95 error 71.21 -> 68.37
mm. Entry mechanics, not a commit or successful unit test, remain the blocker.

V12 IDLE accepts NONE of eight guided trials; output remains unchanged V7.
A separate diagnostic is testing zero EXTRA planning clearance, with original
actual contact envelope, joint/speed guards and acceptance tolerances unchanged.
This isolates whether the additional 1 mm planning buffer contributes to
infeasible local corrections. It is not a relaxed physical acceptance gate or
hardware authorization; no result or adoption claimed yet.

Combined regressions: **191 PASS**; scoped Ruff passes. V12 optimizer evidence
and exact source snapshots: `C:/Users/camer/sonic23_sim_artifacts/`
`pd_shooting_20260908_v12{,_evidence}/` (two optimizer reports, 109 source
bindings archived before further changes). Verification and entry-loading
reports are saved beside those results. Broader held-out dances, causal learned
teleop, real estimator and fresh supervised hardware tests remain required.

## 2026-09-08 Sydney, 11:10 UTC: actual-state substep solver verified; entry unloading defect measured; guided full pilots running

SIM only. No SONIC policy training/change, hardware/DDS/motor command, mode
change, dataset promotion, commit or push. Goal ACTIVE, NOT deployment-ready.

V9 IDLE failure diagnosis reproduced the actual rejected entry: at control
412 the cached state-shifted model predicted +0.721 mm hand/hip clearance,
while the actual next pose reached -0.778 mm. This is a concrete prediction
error, not proof that joint-count mapping alone fixes closed-loop tracking.

Implemented actual-state contact prediction using private copies of complete
integration state including warmstart. All ten original 2 ms PD substeps are
predicted and rechecked. Near-contact repairs differentiate the actual held-
target dynamics; no injected forces or physics changes. V10 BODY completes
2461 controls, cost 8479.93236 versus common V7 baseline 9060.81547. Exact
independent replay verifies all 13 arrays, original full objective, source,
self-contact and all four terminal nonregression checks. Ankles 62.634/40.467
mm: left still FAILS 50 mm. Self hand/hand depth improves from V9 1.178 mm to
0.459 mm, no new penetrating pairs or non-foot ground penetration. Loaded
contact remains unqualified. V10 IDLE accepts none of 24 trials.

V10 exposed numerical clearance stalls and contact corrections that could
violate physical joint/speed bounds. Four original-codec regression cases
reproduced a 20 nm shortfall repeated indefinitely. Fixed by requesting extra
outward planning clearance, NOT widening the acceptance tolerance. Added
original joint/speed inequalities at every predicted substep, with STRICTER
1e-5 rad / rad/s planning insets. Native position and velocity derivatives
match independent physical probes; intermediate-substep limit tests pass.

V11 BODY completes and independently replays exactly: cost 8484.24147;
ankles 62.634/40.467 mm, hands 61.081/70.871 mm, head 35.572 mm. Final joint/
root-position/speed/angle: 0.106126 rad / 44.240 mm / 0.028966 m/s / 0.044278
rad. It is NOT a tracking breakthrough versus V9/V10. Hand penetration now
0.458 mm; peak foot forces 758.60/639.29 N remain recorded, not qualified.
Larger full BODY steps improve cost but regress return speed and are rejected.
V11 IDLE accepts none of eight regularization-10 trials. Exact-state diagnosis
reproduces infeasible correction at control 454; deep/fast-approaching contacts
occur in a rejected private prediction, NOT in accepted execution.

New entry-loading diagnosis independently reproduces 619 complete integration
states of V9 BODY and observes every 500 Hz contact during all 369 entry
controls. At requested first lift apex, ankle target 80.167 mm, actual 38.278
mm. Left foot still carries 59.04 N minimum control-mean and 71.93 N average
during requested flight; none of 19 requested-flight controls unload below
the diagnostic 5 N reporting level. Entry ends with left ankle +54.47 mm X
error. Right foot also stays loaded, 25.24 N minimum / 79.65 N mean, and ends
+23.23 mm X error. COM XY p95 error 71.21 mm; second-apex COM Y error -60.97
mm. These are prefix diagnoses, not complete-motion or contact qualification.

Implemented original-reference transition guidance: additional COM XY and
under-lift terms in LOCAL SEARCH quadratics for entry/return only. Source
frames, timing, cached physical A/B matrices, full acceptance objective and
all nonregression/physical gates stay unchanged. Guidance is not a force,
policy or training-parent promotion. Native finite-difference gradient tests
pass. V12 BODY/IDLE full pilots RUNNING with this guidance and regularization
10 in `pd_shooting_20260908_v12/`; no result claimed yet.

Combined pre-guidance regressions: **185 PASS**. Guidance/protected/shooting
targeted set: **24 PASS**. Scoped Ruff passes. New combined suite running.
V10 archived two optimizer reports / 107 source bindings before V11 changes.
V11 archived five reports/diagnoses / 111 source bindings before guidance:
`C:/Users/camer/sonic23_sim_artifacts/pd_shooting_20260908_v11_evidence/`.
General held-out dances, original29 fidelity, causal SONIC training/teleop,
standing/contact/slip qualification and fresh supervised hardware remain.

## 2026-09-08 Sydney, 10:28 UTC: V9 BODY improves and independently replays; IDLE reactive contact prediction fails

SIM-only. Native SONIC policy unchanged; no robot/DDS, hardware mode changes,
training-parent promotion, commit or push. No deployment readiness claim.

Implemented a coupled inequality-constrained backward solve, including the
state derivative of next-pose self-contact constraints. Clarabel initialization
is independently polished and checked for primal feasibility/stationarity.
V8 BODY comparisons used the SAME cached nominal, derivatives, source, timing,
physics and fixed V7-final protection envelope. Both complete 2461 controls:
box cost 8969.71405, contact cost 8913.66636, versus V7 9060.81547. Both fresh
replays match all 13 physical/integration arrays. Neither passes the source
screen. Only two nominal controls activate self-contact inequalities, while
larger nonlinear updates encounter contacts outside that nominal prediction.

V9 additionally re-solves constraints at each actual simulated state using a
state-shifted next-pose prediction and a STRICTER 1 mm planning buffer. Physical
contact tolerances are unchanged. Acceptance now also rejects regression in
all four final standing metrics, with independent replay verification.

BODY accepts regularization 10, alpha 1/16: cost 8484.76353; ankles p95
62.65/40.44 mm, hands 60.89/69.47 mm, head 35.57 mm. All 2461 controls and
1173 source controls complete. Final joint/root-position/root-speed/root-angle
metrics improve to 0.10610 rad / 44.24 mm / 0.02896 m/s / 0.04430 rad.
13 physical/integration arrays and full objective independently reproduce
exactly; source, self-contact and all four terminal nonregression checks pass.
Self hand/hand penetration decreases 1.458 -> 1.178 mm; no new penetrating self
pairs or non-foot ground contacts. Loaded self-contact still occurs during
486 physics substeps; this is NOT collision qualification. Peak foot forces
759.40/634.78 N are recorded, not qualified. Left ankle still FAILS 50 mm.
The larger alpha 1/8 candidate lowered cost further but worsened final root
speed; it was rejected rather than promoted.

IDLE accepts NONE of 24 reactive updates. Every trial hits the unchanged
right-hand/right-hip self-contact guard during entry (controls ~403-414).
Default output is unchanged V7, not a successful V9 candidate. Its ankles
85.58/80.31 mm and left hand 106.27 mm still fail; return speed 0.08027 m/s
remains unqualified. Next diagnosis compares state-shifted endpoint predictions
with actual 2 ms contact geometry, to distinguish stale dynamics from missed
substep contacts before changing the solver. No unchanged PPO/full-cache loop.

Combined regressions: **168 PASS**. Current experiments are full-future offline
trajectory optimizers, NOT causal SONIC/live teleop. Feasible entry/return,
source fidelity, contact/slip, held-out dance coverage, causal policy training
and fresh supervised hardware evidence remain required. Goal ACTIVE.

Evidence: `C:/Users/camer/sonic23_sim_artifacts/pd_shooting_20260908_v9/`.
V8 archived four reports / 106 source bindings before V9 changes. V9 archived
three reports / 107 source bindings in
`pd_shooting_20260908_v9_evidence/source_archive.json` before further repairs.

## 2026-09-08 Sydney, 09:46 UTC: V7 full pilots independently replay; original-codec rounding defect fixed; source tracking still FAILS

SIM-only progress. No policy training, hardware/DDS/motor command, mode change,
dataset promotion, commit or push. The native SONIC policy itself is unchanged.
Both V7 clips completed two full optimization iterations, with unchanged original
motion/timing, native physics, PD and physical limits. Independent fresh replay
matches all 13 physical/full-integration arrays and full objective bit-exactly.
Per-source-landmark nonregression, every 500 Hz contact check, and final contact
geometry checks pass. These are research nonregression tests, NOT readiness.

| Full clip | V6 source ankles p95 | V7 source ankles p95 | Source screen |
| --- | --- | --- | --- |
| Body, 2461 controls | 64.15 / 44.73 mm | 63.73 / 41.91 mm | FAIL left ankle |
| Idle, 2029 controls | 88.98 / 83.47 mm | 85.58 / 80.31 mm | FAIL both ankles, left hand |

Body hands/head p95 63.36/71.44/36.87 mm; idle 106.27/65.48/70.88 mm.
Protected objective body 9794.09777 -> 9204.85583 -> 9060.81547; idle
37960.89645 -> 33896.60657 -> 32020.76994. Original thresholds stay 50/50/100/
100/100 mm. Both still fail; lower scalar cost must not promote either candidate.

Body existing hand/hand penetration improves 1.647 -> 1.458 mm, no new pairs or
non-foot ground penetration. Idle all four hand/hip pairs become shallower,
worst 1.107 -> 0.926 mm; no new pairs or non-foot ground penetration.
Body final joint/root-position/root-speed: 0.11096 rad / 47.28 mm / 0.02939 m/s.
Idle: 0.11630 rad / 49.75 mm / 0.08027 m/s. **Idle return speed REGRESSES**
from V6 0.06371 m/s; right-foot peak normal force increases 347.51 -> 611.90 N.
No standing/contact/robustness qualification is inferred from source improvement.
Body peak effort still reaches original saturation; no limit was raised.

New diagnosis: V6 body left-foot error is mainly a planted horizontal offset,
median +55.98 mm X, not root drift. First requested entry ankle apex 80.17 mm
is measured only 38.21 mm body / 36.98 mm idle. Entry placement/unloading remains
unsolved; same-reference full-source passes from a source-initialized state
remain prior diagnostics, not substitutes for standing entry.

A separate zero-update test exposed non-idempotent target re-encoding:
11401 of 46667 idle command values changed, max 2.3841858e-7 rad. With zero
feedback, that alone eventually triggered the original height/tilt stop.
With feedback it completed but changed targets up to 4.64916e-6 rad.
Implemented `g1_true23_pd_target_lattice.py`: binary search over original
float32 raw-codec values, returning a nearest actually emitted target plus
forward-codec witness. Original source decoder/transform/bounds are unchanged.
All 2029 original idle commands now survive re-encoding exactly; zero-update
full replay is bit-exact across all 13 arrays both with and without feedback.
This fixes the optimizer's no-change semantics, NOT open-loop robustness or
SONIC policy quality. It is now used by the main offline optimizer.
Combined regressions: **146 PASS**, scoped Ruff passes.

V7 evidence: `C:/Users/camer/sonic23_sim_artifacts/pd_shooting_20260908_v7/`.
Four reports and 103 historical source bindings were preserved BEFORE the
codec integration in `pd_shooting_20260908_v7_evidence/source_archive.json`.
Each V7 cache is hash-bound to its actual nominal trace (iteration 2 caches
linearize iteration 1, NOT the final iteration 2 candidate). Reuse these only
with matching nominal/objective/physics and explicit historical source audit.

Next engineering action: avoid another unchanged full-horizon/PPO loop.
Use cached dynamics to test contact inequalities directly inside the local
constrained control solve (including state-dependent feedback constraints),
with the same full nonlinear physical referee. Large updates are currently
rejected for new/deeper self-contact; accepted body steps have shrunk to 1/128
and only 0.134 mm further left-ankle improvement. Then solve actual entry
weight transfer/lift and terminal settling; preserve every source frame.
No training-parent promotion until physically integrated fidelity, contact
and return evidence supports it. General held-out dance coverage, original29
fidelity, causal learned teleop, live estimator and fresh supervised hardware
tests all remain required. Goal ACTIVE, not complete or blocked.

## 2026-09-08 Sydney, 09:06 UTC: both protected full pilots independently reproduce; numerical box-solver stall fixed; V7 running

Meaningful SIM progress, not deployment readiness. V6 body and idle completed
all 2461/2029 controls, using unchanged physical model, PD, source frames and
timestamps. Independent fresh replays match all 13 physical/full-integration
arrays and the full protected objective exactly. Independent per-landmark
nonregression and self-contact-envelope checks pass, including final geometry.
No learned policy or physical robot changed; no training-parent promotion.

Body protected cost 10480.63861 -> 9794.09777. Source p95 ankle errors
64.15/44.73 mm, hands 65.72/74.24 mm, head 38.59 mm. Left ankle still FAILS
50 mm. Existing hand/hand penetration reduced 1.884 -> 1.647 mm; no new self
pairs or non-foot ground penetration. Peak motor speed 54.33% and effort 100%
of limits are not a safety qualification. Final standing joint error 0.11761
rad, root position error 49.46 mm and root speed 0.03252 m/s remain unqualified.

Idle protected cost 48447.08267 -> 37960.89645. Source ankles 88.98/83.47 mm,
left hand 107.11 mm still FAIL 50/50/100 mm; right hand 67.21 mm and head
73.55 mm. All four existing hand/hip penetrations become shallower, deepest
1.107 mm (previous 1.392); no new self pairs or non-foot ground penetration.
Peak speed 19.94%, effort 84.00%. Final standing root speed 0.06371 m/s is
recorded, not dismissed because the source objective improved.

Saved V6 local quadratics isolated three box active-set failures: free KKT
residuals were already ~1e-16 after Hessian normalization, but ill-conditioning
amplified roundoff into >1e-11 Newton corrections. Fixed the stopping test to
use matrix-product roundoff stationarity; tightened bound multiplier release
to the same tolerance. All four complete cached body/idle backward passes at
regularization 1 and 10 now finish. Three actual failing Hessians are regression
fixtures; independent absolute free solves and KKT sign checks pass.

Primary protected optimizer now checks final post-step geometry before every
acceptance and adds hash-bound linearization/nominal-trace provenance. A caught
trial exception explicitly clears acceptance. Tests reject the actual deep
body collision both at the first physical step and at terminal-only geometry.
Combined regressions: **131 PASS**, Ruff passes the five new/updated files.
One earlier test invocation omitted the existing asset-root environment and
failed model-pin setup; rerunning against the pinned original asset is the
131-pass result. No physical XML or expected hash was altered.

Exact V6 source/report archive (four reports, 103 source bindings):
`C:/Users/camer/sonic23_sim_artifacts/pd_shooting_20260908_v6_evidence/source_archive.json`.
V6 outputs: `pd_shooting_20260908_v6/{body,idle}/{protected,verification}/`.
V7 starts from independently reproduced V6 trajectories, runs two further
protected iterations per full clip, and caches new nominal derivatives:
`pd_shooting_20260908_v7/{body,idle}/protected/`. Both RUNNING, not yet scored.

## 2026-09-08 Sydney, 08:45 UTC: protected full-motion optimizer implemented; both complete pilots running

Previous goal turn was PROGRESS: coupled bound-solver correction had a measured
effect, and independent contact replay changed the next action by rejecting
the later body candidate. This turn preserves the full SONIC/teleop objective.
No hardware actuation, policy training, dataset promotion, commit or push.

Primary `optimize_g1_true23_pd_trajectory.py` now uses the verified coupled box
solve (not clipped unconstrained steps), eight backtracking fractions down to
1/128, and optional saved full linearizations. Default objective is protected:
the original squared motion objective plus per-source-landmark excess penalties
at the UNCHANGED 50/50/100/100/100 mm thresholds, and a 1 mm self-clearance
planning target. These penalties are not safety or deployment certificates.
Accepted full rollouts may not push any initially passing source landmark over
its threshold or worsen an initially failing landmark beyond its fixed starting
p95. Every original frame, phase and timestamp remains in the scoring request.

New `ProtectedPdPlant` checks actual original-model contacts at every 500 Hz
step: non-foot ground penetration is rejected; new penetrating self-contact
pairs and deeper existing self-contacts are rejected against the fixed measured
initialization envelope (1 micrometre comparison tolerance). Existing contacts
are a nonregression research allowance, NOT collision qualification or robot
contact authorization. Severe self-penetration >5 mm cannot initialize this
repair. Actual integration, PD, effort/velocity/joint limits and collision
settings remain unchanged. The rejected V4 body trajectory is not resumed.

Private near-contact geometry queries reuse the existing self-collision helper.
A native deep-hand-contact fixture exposed a reproducible 2.7 ppm bias in its
mesh-normal cost gradient. Active self-distance gradients now use direct native
joint geometry differences at 1e-6 rad; the original strict gradient test passes
without raising its tolerance. No physical collision tolerance was modified.
The actual rejected pose is also rejected at the first physics step in tests.

New explicit historical-source verification authenticates archived Python bytes
while requiring motion/trace data to remain unchanged. It does NOT pretend old
code is current code. Every imported initialization is then independently
replayed against current unchanged physics: all 13 physical/full-integration
arrays must match bit-for-bit. Tests reject modified data, snapshots and report
provenance. The independent verifier now selects the exact recorded objective
and supports this explicit archived-source provenance.

Combined regressions: **122 tests pass**, Ruff passes all six affected/new Python
files. Both V6 full pilots are RUNNING under
`C:/Users/camer/sonic23_sim_artifacts/pd_shooting_20260908_v6/{body,idle}/protected/`.
Body starts from V3 body: all 24610 steps reproduce exactly; only existing
hand/hand self-contact, deepest -1.884013 mm. Idle starts from V5 box idle: all
20290 steps reproduce exactly; four existing hand/hip pairs, worst -1.391541 mm.
Protected initial costs 10480.63861 body / 48447.08267 idle are a DIFFERENT
objective from prior unprotected costs and must not be compared as policy gains.
No new successful tracking result is claimed before complete trial evaluation.

## 2026-09-08 Sydney, 08:16 UTC: bounded-step defect corrected in cached path; later body candidate REJECTED for hand/collision regressions

Actual progress, but NOT SONIC or deployment readiness. No learned policy,
training dataset, physical robot, interlock, git commit or remote was changed.
All research candidates still fail the original full-source landmark screen.

Idle diagnosis located the larger-step failure at control 493: motor speed
ratio 1.09097, while positions were still within limits. Sampled actual-engine
directional derivatives agreed with independent complete-control differences
(relative errors about 1e-6). Zero-alpha feedback completed but was not bit
exact because recanonicalizing original float32 targets adds tiny roundoff;
this small effect did not explain the larger candidate failure.

Alpha 0.03125 completed all 2029 controls with the original clipped quadratic
step: cost 11820.81447 -> 11186.99881, but peak motor speed rose to 90.03% of
limit. Implemented `utils/g1_true23_pd_box_step.py`: a coupled bound-constrained
active-set quadratic solve and free-variable feedback derivative, instead of
clipping an unconstrained step without re-solving remaining joints. Independent
constrained-optimization, active-set derivative and dense full-horizon tests
pass. On the IDENTICAL cached idle dynamics, reference, alpha and regularization,
the proper box solve gives cost **11071.16843**, peak speed **19.47%** of limit
and effort 83.58% (near original baseline 19.58% / 83.48%). Full independent
replay is bit exact across all 13 physical/full-integration arrays.

This is only modest tracking improvement: idle ankle p95 93.81/85.63 mm and
left hand 107.60 mm still FAIL 50/50/100 mm thresholds. Final standing joint
error 0.12977 rad and root position error 73.07 mm remain unqualified. The
original primary optimizer still uses clipped steps; only the explicit cached
`materialize_g1_true23_pd_cached_step.py --step-method box` path uses the fix.
Do not claim the next optimizer integration is already implemented.

Body V4 finished two further iterations: cost 8000.30 -> 7116.99 -> 5474.99.
Final ankle p95 45.60/36.00 mm passes those TWO screens, but left hand worsened
to **157.98 mm**. Independent bit-exact replay exposed new torso/upper-arm and
other self-contacts: loaded penetration up to **23.69 mm**, torso/left-upper-arm
normal force 358.80 N, and peak left-foot normal force 902.89 N. Zero non-foot
ground support does NOT make this safe. **REJECT V4 body as a repair/training
seed.** Scalar average-cost reduction traded away hand accuracy/contact quality.

Retain V3 body only as an unqualified research initialization (left ankle still
65.11 mm; other original source landmark screens pass), and V5 box idle only
as an unqualified research initialization. Neither is an admissible training
parent. No new full-length optimization job remains running at this entry.
Next revision must use the verified box solve, a finer bounded line search,
per-landmark tracking protection and collision constraints. Import any seed by
independent full physical replay; never resume V4 body just because cost is low.

Evidence bundle: `C:/Users/camer/sonic23_sim_artifacts/pd_shooting_20260908_summary_v1/`
contains `decision.json` and `source_archive.json`: 13 report hashes and 104
verified historical Python source bindings. The previous materializer source
was archived before adding the box option, preserving its earlier verified
clipped-step result. Final combined regression: **110 tests pass**; Ruff passes
all 11 new Python files. Source/collision/standing/generalization/live teleop
qualification remains false. See decision JSON for exact trajectories and hashes.

## 2026-09-08 Sydney, 07:40 UTC: first full physical improvement verified; idle pilot rejected

V3 full-body offline trajectory optimization accepted alpha=0.125 after three
larger candidates violated unchanged physical joint bounds. Full original
2461-control lifecycle completed. Objective fell 10109.33699919204 to
8000.30156255978 (20.86%). Source ankle p95: left 71.59 -> 65.11 mm, right
50.94 -> 47.13 mm. Hands/head also improved. LEFT ANKLE STILL FAILS the unchanged
50 mm provisional screen. Final proof joint error 0.13149 -> 0.11609 rad and
root position error 67.37 -> 57.08 mm, but root speed/orientation worsened to
0.01971 m/s / 0.04626 rad. This is partial improvement, NOT lifecycle readiness.
No physical joint-limit excess; peak motor velocity ratio 0.17236 and applied
effort ratio 0.51733. No original reference, timing, physics, gains or bounds
were edited. No new learned policy or hardware run was produced.

Independent complete command replay reproduced all 13 physical/full-integration
arrays bit-for-bit and exactly reproduced full-reference objective 8000.30156.
Observed all 24610 physical substeps: zero non-foot ground support; 857 loaded
hand-to-hand contact substeps, peak normal force 30.54 N. This must be checked
against intended motion contact and is NOT certified collision-free execution.
Maximum loaded penetration 2.60 mm. Original collision masks remain unchanged;
coverage/contact timing/foot slip remain unqualified.

Idle V3 rejected ALL 15 combinations (regularization 1/10/100, alpha
1/.5/.25/.125/.0625) on unchanged joint bounds. No idle improvement accepted.
Reported final trajectory remains the original full baseline, not any failed
prefix. Its independent 20290-substep replay and objective also match exactly.

V3 artifacts: `C:/Users/camer/sonic23_sim_artifacts/pd_shooting_20260908_v3/`
under `{body,idle}/{optimize,verification}/`. Current source checks: 87 combined
regressions plus 3 evidence/contact-observer tests pass; Ruff passes. Reading
solved contacts is itself tested not to alter any integrated state or force.
Added `scripts/verify_g1_true23_pd_trajectory.py` and
`tests/test_g1_true23_pd_trajectory_evidence.py` under `gear_sonic/`.

V4 body is RUNNING two further offline iterations from the verified V3 output,
using exactly the same optimizer/objective and original reference. A separate
SIM-only `scripts/diagnose_g1_true23_pd_feedback_step.py` is RUNNING on full idle:
it caches unchanged full-engine derivatives, checks them against independent
physical finite differences, tests zero-alpha feedback, and logs the exact
joint/control location of rejected trials. Neither run permits weakened limits
or scores a failed prefix as successful. No further result is claimed yet.

## 2026-09-08 Sydney, 07:21 UTC: actual-physics trajectory repair implemented; full-length pilots running

No physical robot operation, new PPO training, commit or push. The rejected V6
free-contact-force controller is not being resumed. Implemented a separate
offline bounded-PD trajectory optimizer, initialized from the complete nominal
policy1300 simulations. It minimizes errors against the UNCHANGED requested
motion and full stand/source/return/standing lifecycle. This is reference
preparation research using future motion, NOT a learned SONIC policy, causal
teleop controller, admissible training parent or deployment qualification.

Completed foundation: the new plant reproduces all 12 original physical arrays
bit-for-bit across every substep of both complete nominal trajectories (body
2461 controls, idle 2029). Full integration states include solver warmstart.
Compiled native23 physics remains
`80c82f5374bed69423580b4d2085e2103bfd83b8691dfe1f7699d2434e1561a8`.
Initial reuse of the earlier QP helper's extra 2-microradian inward target
guard rejected legitimate original-codec endpoint targets. The NEW plant now
uses the exact original composed float32 codec endpoints; neither original
codec nor physical limits changed. Both complete replays then passed.

Eight focused tests pass: exact state replay, original codec endpoints,
external-force/bound rejection, full-control derivatives in flight/contact,
motion-objective gradient including rotated root, zero-feedback physical
identity, and Riccati backward pass against an independently condensed dense
full-horizon quadratic solution. Baseline world-landmark errors also match
the existing referee to 1e-12 m on every source/lifecycle sample in both clips.

New files: `utils/g1_true23_pd_shooting.py`,
`utils/g1_true23_pd_trajectory_optimizer.py`,
`scripts/reproduce_g1_true23_pd_shooting.py`,
`scripts/optimize_g1_true23_pd_trajectory.py`, and
`tests/test_g1_true23_pd_shooting.py` under `gear_sonic/`.
Original 500 Hz PD, 50 Hz targets, motor saturation, joint/velocity limits,
contact solver and source timing remain unchanged. No fictitious root forces,
feedforward torque, mid-trajectory state resets or edited scoring reference.
Each nonlinear candidate must complete the entire lifecycle; falls and physical
joint-bound violations are rejected, not scored as successful short prefixes.
Lower full cost alone does not qualify original-source tracking or standing.

Replay evidence: `C:/Users/camer/sonic23_sim_artifacts/pd_shooting_20260908_v2/`
under `{body,idle}/baseline/`. Body trace SHA256
`43f8ab5567e714f50c48f9398a424fb2edb1fe109d0987c505d2a4551e7c304a`;
idle `357fed09c6182d55a3ec2809f6fbcc395f177d440879c52010f86242191760e5`.
Full one-iteration pilots are RUNNING at this entry's timestamp under
`pd_shooting_20260908_v3/{body,idle}/optimize/`; no optimization improvement
is claimed until their complete physical rollouts and source metrics finish.

## 2026-09-08 Sydney, 06:35 UTC: reference/contact reassessment complete; no torque or tempo shortcut demonstrated

Completed the agreed bounded experiment AND follow-up reference feasibility
audit. V6 controller branch remains REJECTED: body and idle both fell, as
recorded below. No new training, controller variant, robot run, commit or push.
Combined regression run: **98 tests pass**; Ruff passes for all five affected
Python files. Software checks are not motion or deployment qualification.

Added `gear_sonic/scripts/audit_g1_true23_pd_reference_authority.py` and 15
focused tests. It compares identical candidate contact cones, full native23
force balance and optimistic bounded friction under motor-only versus reachable
PD-target torque bounds. All original reference command samples are checked;
unknown solver outcomes remain separate. Independent force/bound checks are
required. A verified feasible witness in a containing torque envelope prevents
contradictory paired classifications; no tolerance is relaxed. Archived source
snapshots and all input hashes were verified in the final reanalysis.

Important calibration: the exact initial standing reference has toe corners
4.0225 mm above the plane. A 4 mm candidate cutoff keeps only four heel contacts
and incorrectly suggests that exact reference has no support, despite the
successful actual standing simulation. A 5 mm OFFLINE candidate hypothesis
admits all eight corners and makes the standing references conditionally
feasible. Neither hypothesis changes actual physical collision margins. The
audit explicitly CANNOT be used as an automatic dataset rejection gate, actual
contact proof, or a claim that approximate motion is physically impossible.

The PD restriction persists under BOTH contact assumptions: 74 body and 63 idle
original command samples are motor-conditionally-feasible but PD-infeasible.
Under the 5 mm hypothesis these split as follows:
- body: 32 entry, 6 source, 36 return;
- idle: 24 entry, 5 source, 34 return.
Other support failures remain, including source samples. This points strongly
at transition design without falsely declaring source tracking solved.

Uniform 2x-duration OFFLINE timing hypothesis reduces extra PD restrictions
to 5 body / 10 idle samples, but increases motor-only support infeasibility from
67 to 113 body samples and 43 to 98 idle samples. Required balancing acceleration
changes when timing changes. Slowing every pose is therefore not an established
repair. No 50 Hz retimed rollout, between-sample qualification or slower-policy
success is claimed; source poses themselves were not edited.

Reassessment bundle:
`C:/Users/camer/sonic23_sim_artifacts/pd_reference_authority_20260908_reassessment/`
contains `report.json`, `verification.json`, the archived-audit comparison script,
and final audit source snapshot. Report SHA256:
`abfabac49dbf61439eacb12010feea5eeebb202983761466b3a18b4064cf2e11`.
Original hypothesis reports remain in `pd_reference_authority_20260908_v1`,
`pd_reference_authority_20260908_v2_gap5` and `pd_reference_authority_20260908_v3_duration2`.

Next required repair is NOT implemented: optimize native23 pose/contact/timing
trajectories together, including realizable PD torque and balance, instead of
geometry-first references plus a motor-effort-only screen. Require actual
uninterrupted SIM entry, full source, return and standing proof on those
references before claiming an executable motion or resuming broad SONIC training.
Keep original 29-DOF motion intent and all source samples, but explicitly report
necessary feasible adaptations; universal exact 29-DOF reproduction is not a
valid promise for a robot missing six joints. The native23 SONIC deployment goal
remains UNMET. No new full dance or teleop pass exists.

## 2026-09-08 Sydney, 06:06 UTC: bounded torque experiment FAILED its decision gate; controller branch rejected

V6 is finished, not pending. Thirty-one focused tests and Ruff pass. Zero
strength completes the full 2461-control body replay with all 38 inherited
trace arrays BIT-EXACT to policy1300, plus identical physics, lifecycle and
policy identities. Zero trace SHA256:
`46995e4550235292b4a16bcf234b4105c2b64330b017dc0af76386d19f420fe8`.

The 250-control standing smoke passes without saturation or controller failure:
root p95 4.1147 mm, ankles 5.7406/5.7343 mm. This is only five seconds standing.
Standing trace SHA256:
`2986d3af763e9e513b792b8aaf32375ea869e865d3c35a0e53f532a5e1f6d707`.

BOTH full nominal motion tests FAILED with actual falls, not solver tolerance
or controller-input errors. Body integrates 965/2461 controls (346/1173 source
controls); idle 608/2029 (0/741 source controls). Both hit the unchanged absolute
height/tilt diagnostic stop. Neither reaches return-to-standing. Minimum pelvis
heights are 0.11382/0.11945 m. Source frames/timing and acceptance tests were
not shortened or changed. Body/idle trace SHA256:
`47309bca54f3fa70907ae78d52123e6c74a72e49ad7d2f230d7c40b7101f0a9a`,
`5b3bd049ffacaa7ac429afdfa607221a63fe34c71575ab2db09d922176dc897c`.

The torque bottleneck is real but NOT the entire cause: body first landing
root error at control 410 drops from 77.25 mm (V5) to 7.30 mm (V6). However,
right-foot error is still 76.36 mm at source entry. Stance/foot tracking and
balance remain insufficient; extra torque alone does not solve the task.
Candidate contact forces are not verified real contact dynamics. At body
control 900 the QP predicts root-x acceleration -0.2107 m/s^2 while the first
actual physical substep gives +0.1099 m/s^2. This is evidence of remaining
model/controller mismatch, not a proven single root cause.

A saved-trace audit independently recomputes EVERY requested PD+feedforward
torque, saturation and engine actuator force BIT-EXACT. Feedforward is held
for exactly ten substeps, targets retain the original envelope, actual total
torque never exceeds configured effort, and no external pelvis force is used.
Physical compiled hash remains
`80c82f5374bed69423580b4d2085e2103bfd83b8691dfe1f7699d2434e1561a8`.

Decision: REJECT this controller/reference branch. No additional training,
controller tuning, hardware commands or promotion. Existing SONIC policy1300
is unchanged and still fails its six full-motion screens. Deployment goal is
NOT achieved. This result does not prove 23-DOF SONIC impossible.

Durable experiment bundle:
`C:/Users/camer/sonic23_sim_artifacts/inverse_dynamics_tracking_20260908_v6/`
contains `decision.json`, all four evaluations, read-only reproducibility script
`audit_replay.py`, and final controller/runner source snapshots. The justified
reassessment is native23 dynamics/contact-consistent reference preparation,
including reachable PD torque bounds, followed by actual full-lifecycle SIM
qualification before further SONIC training. A missing six-joint output edit
or adding this unqualified controller to hardware is not a demonstrated fix.

## 2026-09-08 Sydney, 05:55 UTC: V5 failed; bounded SIM torque experiment implemented, not yet proven

V5 full nominal runs FAILED: body stopped on QP infeasibility at 730/2461
controls; idle at 592/2029. Idle actually fell (pelvis height about 0.216 m at
control 580). First foot-placement error improved, but balance did not survive.
No full dance or live teleop success. Failed traces and V5 controller/runner
snapshots remain in `inverse_dynamics_tracking_20260908_v5/{body,idle}/full/`.

Measured balancing authority is limited by BOTH motor effort and bounded PD
targets. At body reference control 392 the stance ankle can request about
-2.0523 Nm through the existing PD target envelope, while a motor-limit-only
inverse-dynamics solution needs about -2.7652 Nm. Among 41 sampled first-step
reference controls, 18 body and 19 idle controls are motor-only feasible but
certified infeasible with PD-target torque bounds. Unknown LP outcomes are not
counted as infeasible. This conditional force check uses candidate contacts;
it is NOT a proof of actual contact, sole failure cause, or trajectory feasibility.

V6 is an explicitly alternate SIM actuator law: existing bounded 23-joint
position targets PLUS held joint feedforward, with both feedforward magnitude
and total applied torque bounded by the same configured motor efforts. Physical
model, gains, joint limits, source frames, reference timing and acceptance tests
are unchanged. No root/ground forces are injected. It is a privileged model-based
counterfactual, NOT pure SONIC; existing SONIC previous-action history does not
encode feedforward. No hardware transport or deployment authorization exists.

The new wrapper records every physical feedforward request, expires each command
after ten 500 Hz substeps, and rejects stale reuse or a failed fresh QP. Zero
strength must reproduce the original policy replay bit-for-bit. Unit validation
is in progress; no V6 standing or full-motion result yet. Decision gate: full
body/idle lifecycle balance and existing tracking requirements, not more training.
If those fail, reject this controller branch and reassess; no reward-tuning loop.

## 2026-09-08 Sydney, 05:02 UTC: inverse-dynamics entry failure isolated; world-foot task testing

V2 nominal body/idle probes stopped at 428/417 controls on `AlmostSolved`.
Equivalent variable scaling alone (V3) solved those saved states but stopped
both new runs at control 20 on the same precision status. No reduced-accuracy
solution was accepted. Static KKT regularization caused numerical stalling;
an isolated controller backend disables that perturbation but preserves dynamic
pivot protection, iterative refinement, requested primal/dual/gap tolerances
1e-9 and independent ORIGINAL constraint audit 1e-8. The shared offline QP
solver and all bound reference-preparation implementations remain unchanged.
All 177 selected saved states, including all three distinct recorded failure
states, strictly solve with this final backend. Unit tests also reject mocked
`AlmostSolved`, nonfinite and false-success constraint-violating solutions.

V4 then progressed to control 510 in BOTH clips and stopped on true controller
constraint infeasibility. Neither full timeline completed; no source tracking
pass is claimed. At first swing peak, actual left ankle reached 61-65 mm versus
the requested 80 mm, compared with 37-38 mm in pure policy1300. However, first
foot then landed 6-10 cm too far sideways. During second swing, pelvis y error
grew to about -132 mm and the right swing foot hit the floor early. Those actual
contacts are not hidden; the reference-nominated contact model could no longer
produce a solution. No fall was relabeled as tracking, no state reset or
fallback used. V2/V3/V4 failed traces and controller source snapshots remain.

V5 adds explicit world-space acceleration tracking for all eight physical sole
sphere centers, including received-reference velocity/acceleration feedforward.
It can compensate base-position error with leg posture instead of copying that
error into foot placement. Existing root/joint objectives and physical/target/
effort/acceleration bounds, all reference frames/timing, and tracking acceptance
criteria stay fixed. This is a controller objective change, not SONIC learning
or evidence of source parity. Twenty-two focused tests pass, including exact
base-translation and reference-dynamics regressions. Full nominal body and idle
rollouts are running at `inverse_dynamics_tracking_20260908_v5/{body,idle}/full/`.
No V5 outcome yet. SIM only; physical robot untouched.

## 2026-09-08 Sydney, 04:46 UTC: whole-body controller holds standing; full two-clip tests running

Implemented a separate, explicitly privileged SIM-only inverse-dynamics
controller, not SONIC and not an eligible normal campaign parent. It receives
copied measured native23 state and only the already-received q0/q1/q2 poses,
solves unactuated-root dynamics with unilateral physical-sole contact cones,
and converts bounded joint torques to targets through unchanged source/native
codecs and 50 Hz held targets / 500 Hz physical PD. Optimized contact forces
are NEVER applied to the simulator. Physical model, collision geometry, gains,
effort/target bounds, full source and timing are unchanged. Contact hypotheses
are not complementarity or dynamic feasibility proof. No hardware transport.

V1 standing failed at control 25/250. Trace diagnosis: reference corner gating
discarded both toes because the initial reference toe gap is 4.0225 mm while
heel gap is 3.0025 mm. It kept heel-only support even after actual simulator
toes contacted the floor. V2 nominates a stance foot from its minimum reference
gap, then admits individual contacts from MEASURED sole collision geometry;
floating measured corners still cannot supply contact force. No acceptance
threshold or physical margin changed. V1 source snapshots and failed trace
are retained in `inverse_dynamics_tracking_20260908_v1/body/standing/`.

V2 completes all 250 standing controls: root p95 4.4675 mm, ankle p95
5.5875/5.5828 mm, hand p95 4.4499/4.4490 mm, head p95 4.9416 mm; final joint
RMSE 5.9122 mrad; no controller failure, saturation or midrun state reset.
Maximum independently checked force residual 1.71e-13; QP solve p95 1.466 ms,
maximum 1.702 ms (QP time is NOT total controller latency qualification).
Physical compiled hash remains `80c82f5374bed69423580b4d2085e2103bfd83b8691dfe1f7699d2434e1561a8`.
Standing trace SHA256 `0055d74ff969e376fab5df9916d5f8b0827ba74a4fd50bf639b29d91c0b50800`.
This is a prefix diagnostic, not full lifecycle success. Sixteen focused tests
pass, including a real-model regression for retained measured toe contacts.
The V1 zero-strength full body replay independently matches all 38 baseline
trace arrays bit-for-bit, all 2461 controls and lifecycle metrics unchanged;
zero trace SHA256 `80d657e34e0ce273983707a765a7f047712a34b85952895b78b558d1dfcacb1f`.

Both unchanged full nominal body/idle entry-source-return timelines now run
under V2 in `inverse_dynamics_tracking_20260908_v2/{body,idle}/full/`. No full
result yet and no teacher training or deployment promotion authorized by this
entry. The better pure SONIC research baseline remains corrected-step1300.

## 2026-09-08 Sydney, 04:25 UTC: sole-world1400 rejected after all six full cases

All six same-reference/physics cases completed without control failures, falls,
midrun resets or fallback. All six still FAIL source landmarks. Independent
trace comparisons are complete in the sole-world campaign directory. Body
nominal feet 71.593/50.937 -> 65.623/55.730 mm, hands 73.579/83.826 ->
94.194/104.961 mm, head 45.855 -> 68.021 mm. Body final joint error improves
131.489 -> 106.668 mrad but final root worsens 67.374 -> 68.918 mm. Idle feet
97.194/88.273 -> 102.588/131.572 mm; root p95 76.168 -> 129.917 mm and final
root 76.227 -> 130.725 mm. Push regressions also remain visible. The new reward
is a failed research branch, not a reason to continue1400 automatically.

The pure corrected-step policy1300 remains the better research baseline. Two
subsequent learning changes and the swing-only target override have not solved
coupled support/placement. Next bounded diagnostic is a separate SIM-only
full-body inverse-dynamics tracking controller with unchanged physical model,
PD gains and effort/target limits, to test executable balance/entry before more
SONIC learning. It is explicitly not a learned SONIC policy or a deployment
substitute. No code or success of that controller is claimed in this entry;
implementation/verification follows. No hardware commands.

## 2026-09-08 Sydney, 04:17 UTC: sole-world policy1400 trained and verified; six full replays started

The controlled pure-policy1300 -> sole-world-policy1400 branch completed all
100 updates, 819,200 new samples, 10,240,000 cumulative. It used the unchanged
mixed-reference schedule: 98 standing resets, 604 sampled-source resets,
66 complete standing-start training timelines and 508 source suffixes. Suffixes
and noisy training completion are not tracking or deployment qualification.

Independent checkpoint audit passes exact initial actor/critic/optimizer/counter
transfer, ten unchanged frozen encoder tensors, all 18 decoder tensors and all
36,864 root-conditioner elements changed, finite training state and bounded
exploration. Checkpoint SHA256
`30ddb681a684675f0ef0dbdadf0c9a6ac3cbf42b7a4a1df5651f6b9bea8c02c4`,
actor `0dec4fd3d1a38611d551f6bc3b2cd4fa07ad5168bc0f356f21f30c09ee93ea90`,
lineage `2a8e15ce374725d6c4ede580e69f1e03136c067467b469473df92fec15a1d0f0`.
Export parity passes: encoder exact, decoder maximum absolute error 1.1324883e-6;
decoder ONNX `6f38d02fbfc1e3942ee2e73987e4a74124dd0c473ed3e75bcfb27625d38c681d`.

Both clips' full nominal/push-X/push-Y CPU campaigns have started. No quality
result or promotion yet. Evidence directory:
`contact_step_reference_bank_20260908_v1/campaign/segment_1300_1400_sole_world_v1/`.
This is separate from rejected phase-balanced1400. No physical robot commands.

## 2026-09-08 Sydney, 03:59 UTC: swing-only override rejected; coupled sole-tracking training change implemented

Both complete zero-strength controls match all 38 inherited policy1300 trace
arrays bit-for-bit, identical policy bytes, initial state, physics, full reference
and lifecycle metrics. Body 2461 and idle 2029 controls completed. The full
swing-reference override subsequently completed both timelines but FAILS both
source screens. Body ankle p95 worsens to 110.596/177.812 mm and root p95 to
168.299 mm. Idle left improves to 73.935 mm, but right worsens to 108.534 mm,
hands to 151.322/171.260 mm, root p95 to 119.918 mm. No controller promotion,
no training from this hybrid, and no swing-action imitation objective adopted.
Evidence: `swing_reference_target_probe_20260908_v1/{body,idle}/{zero,reference_blend}/`.

At first swing peak the override requests/measures the reference knee near
1.05/1.11 rad, yet left ankle remains at 34.716/35.831 mm, not the requested
80.167 mm. Body/idle pelvis falls to 698.306/697.150 mm. Saved-state FK shows
actual right sole inner edge about y=-108 mm, while measured COM y is only
-62.598/-57.951 mm in the override. The unchanged baseline already spreads feet
roughly 2 cm during initial standing, then incompletely tracks weight transfer.
COM projection alone is not a dynamic stability test; these are measured pose
facts explaining why stronger swing-joint requests did not produce clean lift.

Implemented explicit `root_and_upper_feet_sole_world_v5`: one additional training
cost on actual world positions of all four physical sole-sphere bottom points
per foot (2 cm normalization, weight -1). This penalizes stance displacement,
missing swing clearance and foot orientation together. Existing v4 reward terms,
source arrays, start schedule, frozen SONIC inputs/encoder, actor, gains, physical
limits, and CPU acceptance thresholds are unchanged. No contact-label gating,
measured-state target reanchoring or action override. It does not certify force,
slip or dynamic feasibility. The next controlled branch uses pure policy1300 and
its unchanged mixed-reference reset schedule, not the rejected phase-balanced
policy1400. All 81 focused tests and the expanded 308-test regression suite pass.
The 100-update branch is running at
`contact_step_reference_bank_20260908_v1/campaign/segment_1300_1400_sole_world_v1/`.
Executed GPU configuration confirms 23 actuators, unchanged mixed-reset command,
all 20 prior rewards plus the sole term, and zero initial reset retries/physics
prime steps. As of 04:10 UTC, 63 updates completed; no policy quality result yet.
SIM ONLY.

In parallel, the first frozen-cohort BONES locomotion source,
`walk_hands_on_back_180_start_R_001__A052`, received one unchanged bounded 2x-time,
full-excursion native23 fit. It passes the 469-frame control-grid fit but FAILS
independent all-original-timestamp audit: 17/563 original samples fail (one COM,
five left-foot orientation, eleven right-foot orientation). Maximum extra foot
orientation regression is 0.005549/0.003538 rad. Nominal inverse-dynamics support
audit also finds no solution on 426/469 frames, no missing contact candidates,
and only 43 conditionally feasible frames. It is NOT added to the training bank.
All evidence and rejected candidate remain local at
`bones_seed_walk_retarget_20260908_v1/walk_hands_on_back_180_start_R_001__A052/`.

## 2026-09-08 Sydney, 03:44 UTC: phase-balanced policy1400 rejected; isolated swing-target SIM probe

All six complete policy1400 CPU cases finished without falls, midrun resets,
fallback or control errors, but all six still FAIL the unchanged source screen.
The phase-balanced change is a regression, not a promotion: body nominal feet
71.593/50.937 -> 74.583/46.037 mm; idle 97.194/88.273 -> 113.765/198.479 mm.
Idle root p95 76.168 -> 175.451 mm; final root 76.227 -> 178.155 mm and joint
error 134.408 -> 185.073 mrad. Body final joint error worsens to 174.940 mrad.
Matched comparisons and all original case traces remain in the phase-balanced
campaign directory. Export parity passed (max absolute error 1.4305115e-6),
which did not predict controller quality. Do not resume this policy1400 branch.

The better, still unqualified corrected-step policy1300 remains the research
baseline. Implemented a separate SIM-only counterfactual that blends only
generated-transition swing-leg outputs toward already-received q1 joint poses.
All four reference sole corners must clear the floor before activation; source,
standing, stance leg, waist and arm outputs are not directly overridden.
Original gains, effort limits, target projection, all source frames, initial
state and timing remain unchanged. This is explicitly not a pure learned SONIC
controller and cannot become a normal training parent or deployment receipt.

Twelve focused tests and scoped Ruff pass. Complete zero-strength controls for
both clips are running first; every inherited trace array must match the actual
policy1300 baseline before interpreting a nonzero experiment. The purpose is to
distinguish weak learned foot-lift requests from failure of the proposed
support/reference motion. No probe result or auxiliary-imitation benefit is
claimed yet. SIM ONLY: no robot transport, motor or mode commands.

## 2026-09-08 Sydney, 03:23 UTC: phase-balanced policy1400 trained and verified; full replay pending

The explicit schedule-change block completed 100 updates/819,200 additional
samples, 10,240,000 cumulative. Actual reset counts: standing 99, entry 128,
source 200, return 441. Every entry/source/return decile was sampled. There were
67 completed standing-start training timelines and 673 completed phase suffixes;
suffixes remain explicitly excluded from full-lifecycle qualification. The
executed command is `PhaseReferenceRootFeedbackCommand`; initial checks needed
zero reset retries. Broader focused regression suite passes all 255 tests.

Continuation audit passes: exact parent actor/critic/optimizer/counters at
initialization, ten frozen encoder tensors unchanged, all 18 decoder tensors and
36,864 root-conditioner parameters updated, finite training states and bounded
exploration. Checkpoint SHA256
`dd1a7a5060a0eb974feddfe1bcbee4ee6d5a81bfddc14002c990715d243bc49b`,
lineage `4bf738234c70338808f5170f4c89500d65e47537d3c264c0807b8d8da92963e7`.
Export is underway, then all six same-reference full CPU cases are required.
No policy1400 tracking result or deployment claim yet.

Read-only policy1300 trace audit provides a concrete remaining failure: at first
left swing peak (control 392), idle requests knee 0.690744 rad versus reference
1.112402 rad; measured knee 0.752403 rad and ankle height 36.622 mm versus desired
80.167 mm. Body requests knee 0.673158 versus reference 1.053767 rad. No torque
saturation or meaningful action projection at these controls. Instantaneous FK
of requested joint targets is only a diagnostic, not integrated dynamics.
Both transition plans swing the left foot first. The lower-input signal is not
constant: retaining measured orientation/VR and replacing only swing lower240
with standing lower240 changes 33-42 of the frozen encoder's 64 FSQ components
at the four lift peaks (maximum token change 0.125). Original C++ lower-body
absolute-position packing and joint ordering were rechecked read-only. These
findings support weak learned stepping requests; they do not prove the absence
of every possible semantic defect. SIM ONLY, no robot commands.

## 2026-09-08 Sydney, 03:03 UTC: real but incomplete tracking improvement; phase-balanced learning launched

All six full step-reference policy1300 CPU replays completed without falls,
midrun resets, fallback or control failures. All six still FAIL the unchanged
source landmark screen. Independently recomputed matched comparisons are saved
in `contact_step_reference_bank_20260908_v1/campaign/segment_1200_1300_contact_steps_v1/`.

- Body nominal source ankle p95: 74.059/53.043 -> 71.593/50.937 mm;
  hands 101.884/114.592 -> 73.579/83.826 mm; head 80.094 -> 45.855 mm.
  Root p95 91.087 -> 82.113 mm; final root error 86.731 -> 67.374 mm.
- Idle nominal source ankle p95: 92.835/170.266 -> 97.194/88.273 mm.
  The right foot improves 81.993 mm, but the left worsens 4.359 mm.
  Root p95 130.921 -> 76.168 mm; final root 131.765 -> 76.227 mm.
  Final joint error worsens 130.902 -> 134.408 mrad. Push-case regressions in
  individual feet/return speed remain visible; no mean score hides them.
- Idle left foot is already 90.241 mm off at acquisition end and remains
  105.311 mm off at final standing. Entry/return remain unsolved.

Export passes: frozen encoder exact, decoder numerical parity max absolute
error 8.046627e-7, native output 23. Actor SHA256
`043d08759b4ccfb2148682565d4b89f9e1e03996afce7132a1e3065c261005f8`,
decoder `83c5c6bb71ea96b3dbe6717cec1e0a39e64ee2bd389d082ae7def7fcfebf3c28`.
No production promotion or readiness claim.

The preceding block had 604 sampled-source resets versus 98 standing starts,
with no dedicated entry/return initialization. Implemented explicit
`phase_balanced_reference_reset_v1`: one quarter each standing/entry/source/return.
Only environment-reset-time state initialization changes; every remaining
reference frame executes and sampled suffixes never count as full lifecycles.
All source arrays, physics, optimizer/LRs, objectives and CPU tests are unchanged.
Ninety-three focused scheduler/trainer/bank tests and scoped Ruff pass.

The evaluated policy1300 remains a research parent, not a qualified deployment.
Its next 100-update segment has launched at
`.../campaign/segment_1300_1400_phase_balanced_v1/`, explicitly declaring the
schedule change and binding all six completed parent replays. New GPU progress,
checkpoint verification/export and matched complete replays are still required.
SIM ONLY. General dance coverage, contact dynamics, live estimator and teleop
remain unqualified; no hardware commands were sent.

## 2026-09-08 Sydney, 02:53 UTC: first corrected-step training block completed and verified; export underway

The separate step-reference policy1200-to1300 branch completed 100 PPO updates
and 819,200 new simulation samples (9,420,800 cumulative). Runtime records 98
standing resets, 604 sampled-source resets, 66 completed standing-start training
timelines and 508 completed source suffixes. These noisy training episodes are
NOT deterministic replay or deployment evidence. Complete entry/return/source
references remain unchanged throughout this run.

Independent checkpoint audit passes: initial actor/critic/optimizer/counters
exactly match policy1200; all ten frozen SONIC encoder tensors unchanged; all
18 decoder tensors and 36,864 root-conditioner elements updated; finite critic,
optimizer and bounded exploration. New checkpoint SHA256
`e591f475f1cc173de8d7fd1a1f092a740aa5e87278b69bc30cf82cc8745d0f45`,
new lineage `ed5cc9482584683647d5807a98bc5e3547682a297ad785040d06e2450133a739`.
Evidence in the step-bank campaign's `continuation_verification.json`.

Export is running, followed by both clips' complete nominal/push-X/push-Y CPU
replays. No policy quality result or promotion yet. Matched comparison now
accepts this separately declared generated-transition profile, rejects root-input
counterfactuals and compares independently recomputed per-phase errors as well as
full source/return metrics. Sixteen scoped comparison tests pass. SIM ONLY;
hardware, live estimator, general dance coverage and dynamic feasibility remain
unqualified.

## 2026-09-08 Sydney, 02:34 UTC: corrected step geometry passes; policy1200 still fails all six full replays; explicit step-training branch launched

SIM ONLY. No robot commands, hardware qualification, policy promotion, commit or
push. V4 stepping references retain every original V5 source sample/channel and
all standalone standing states. Generated entry/return are each 369 controls
(7.4 s endpoint-to-endpoint), with lift, fixed support-foot XY, bounded base tilt,
native action bounds, and generated-only arm clearance/rate repair. All four
transitions pass independently recomputed 50/100 Hz geometry, floor and
self-collision checks. Maximum joint speed/acceleration remain within 5 rad/s and
80 rad/s^2. Geometry does NOT establish dynamic feasibility: conditional support
failures remain body entry/return 41/41 and idle 47/42; body return has one
numerically indeterminate frame, idle entry has one effort-limit exceedance.
No unsupported or indeterminate frame is dropped or labelled feasible.

All six unchanged-policy1200 V4 full lifecycle replays completed without falls,
midrun resets or fallback, but FAIL source tracking. Body ankle p95 is
74.059/53.043 mm; idle 92.835/170.266 mm. Nominal root p95 is 91.087/130.921 mm.
Actual left-foot entry travel is only 14.84/11.55 mm versus requested 81.20/79.12
mm. This is a learned tracking/unloading failure after geometric repair. The
separate root-feedback x4 probe still fails both foot screens; no gain sweep or
counterfactual is adopted. Source-initialized passes from 01:48 remain diagnostic
only, not full-lifecycle success.

New explicit training integration binds V4 references and all six policy1200
baseline outcomes, preserving raw source-bank ownership/frame counts/timing and
the original policy1200 actor, critic, optimizer and counters. Source contact
conditioning plus generated transitions are explicitly changed; this is NOT a
same-reference resume. Frozen SONIC encoder, reward, action/physics limits and
training settings remain unchanged. A separate 100-update segment is launched
under `C:/Users/camer/sonic23_sim_artifacts/contact_step_reference_bank_20260908_v1/`
with the prior regressed contact-only1300 excluded as a parent. Launch/preflight
is not training completion; continuation verification, export and all six matched
CPU replays are still required. No new candidate has yet been promoted.

162 existing training/curriculum/reference-bank tests pass. New explicit
step-transition layout, preservation, failure-disclosure and CPU-profile guards
plus campaign tests pass (48 tests); scoped Ruff passes. Test success is not
motion-tracking evidence. V4 reference and baseline artifacts are under
`C:/Users/camer/sonic23_sim_artifacts/contact_step_lifecycle_20260908_v4/`.

## 2026-09-08 Sydney, 01:48 UTC: source-initialized policy passes both source screens; stepping transition implementation under audit

Major diagnostic result, NOT lifecycle/deployment readiness: unchanged policy1200
on unchanged full V5 sources passes the existing landmark screen when initialized
at source frame zero. Only initial standing/acquisition is omitted; all source
frames and the complete old return/standing tail are preserved, no postinitial
reset or fallback. Explicit synthetic initial history and privileged simulator
root measurement remain disclosed. Evidence under
`C:/Users/camer/sonic23_sim_artifacts/source_initialized_20260908_v1/`.

- Body-check: all 1173 source controls (1573 including return) completed.
  Ankle p95 14.128/13.615 mm; hands 89.664/85.344 mm; head 71.125 mm.
  Normal standing-entry baseline ankles were 68.902/54.153 mm.
- Idle: all 741 source controls (1141 including return) completed.
  Ankle p95 8.299/48.357 mm; hands 55.414/81.379 mm; head 99.179 mm.
  Normal standing-entry baseline ankles were 105.033/143.966 mm.

This isolates entry as a major failure, not a general-motion qualification.
Idle remains close to the head/right-foot thresholds. Return joint errors remain
168/159 mrad; the old return is not fixed by initialization. A new SIM-only script
and six tests preserve/label this counterfactual and prevent normal continuation
from accepting it as a complete lifecycle result.

Read-only sole audit confirms old entry ramps slide feet 81/72 mm (body) and
79/168 mm (idle) with no lift beyond the initial 3 mm floor gap. They also contain
up to 2.69 mm and 10.52 mm floor penetration respectively. Raising motor limits
does not address this defect.

Implemented separate native23 contact-step transition generator: fixed support
foot XY, timed swing, edge-roll toe-off/landing, COM trajectory constrained by an
explicit 0.65 m LIPM support polygon, unchanged joint/action limits, bounded base
tilt, unchanged full source and standalone standing states. Static COM transfer
was rejected because of ankle limits; dynamic steps solve all four endpoint
problems. Initial V1 full-path audits still failed arm collisions, leg acceleration
and sparse 100 Hz floor clearance; those references were not policy-tested.
Actual support failures V1 body entry/return 46/69, idle 48/60 remain recorded.

V2 preparation is running under `contact_step_lifecycle_20260908_v2`: generated
swings lengthened from 0.5 to 0.7 s only, target floor gap 0.5 mm, existing bounded
multi-arm clearance/rate repair applied with its unchanged 1 mm COM budget.
Source timing is NOT changed. No new learned-policy candidate or training run.
New model-backed tests initially exposed the already-known working-copy raw XML
line-ending hash mismatch; their asset root is now explicit, using the same pinned
original assets as CPU replay. No model checksum or geometry was changed.

## 2026-09-08 Sydney, 01:12 UTC: policy1300 rejected; entry transition now isolated as a major error source

SIM ONLY. All six complete policy1300 replays failed the unchanged tracking
screen. No promotion, hardware action, commit or push. Body nominal root p95
regressed from 46.863 to 61.521 mm; right ankle from 54.153 to 59.366 mm.
Idle right ankle regressed from 143.966 to 151.367 mm. Per-landmark, same-reference
comparisons are saved under the contact-bank segment; no improving average hides
these regressions. The frozen encoder and export parity checks still pass, but
those are not evidence of useful motion tracking.

Explicit SIM counterfactuals also completed: multiplying only root-position
feedback by four improved root drift, but idle ankle p95 remained 106.654/142.222
mm. Adding bounded measured-leg position feedback worsened motion tracking.
Both probes remain separately labelled, ineligible for normal continuation or
deployment. Their zero-change controls reproduced every baseline trace array
bit-exactly. Scoped comparison/root/leg tests pass (9/10/9 respectively).

Important phase diagnosis: policy1200 standing ankle p95 is 20.34/15.28 mm.
At acquisition end, BEFORE body-check starts, errors are already 64.44/56.62 mm;
before idle starts, 88.67/147.07 mm. Body-check source feet barely move thereafter.
Generated entry asks feet to move about 81/72 mm for body-check and 79/168 mm for
idle. Idle has zero target projection and zero requested-torque clipping; body
has only small, rare ankle projection and no torque clipping. Raising actuator
limits is therefore unsupported. Next: explicitly source-initialized replay to
separate entry failure from intrinsic tracking failure; then contact-aware
stepping transitions if supported. Such replay is NOT standing acquisition or
full-lifecycle qualification, and source samples must remain complete/unmodified.

The full planned-dance contact correction completed: 1091 controls, 2181 half-grid
samples, zero self-collision, minimum sole gap 0.099969 mm. However conditional
force-support failures remain 960/1091 (was 1049); no-contact frames fell from 555
to 24. Geometry improved, dynamics still fail. Dance remains outside the accepted
training bank. Artifact `contact_conditioned_reference_20260908_v6_dance/planned_full_dance`.

## 2026-09-08 Sydney, 00:33 UTC: contact-conditioned policy1300 trained, verified and exported; CPU replay underway

The explicit contact-reference segment completed 100 updates, 819,200 new
simulation samples and 9,420,800 cumulative samples. This is the first learning
run on these corrected sources, not another unchanged-reference continuation.
No physical robot, deployment edits, commits or pushes.

Actual continuation checks PASS: initial actor, critic, optimizer and counters
exactly match policy1200; all 10 encoder tensors unchanged; all 18 decoder tensors
and all 36,864 conditioner parameters updated; critic/optimizer finite. Gaussian
standard deviation range 0.099992856..0.100039840. Nominal physics, rewards,
learning rates, initial-state schedule and frozen SONIC encoder unchanged.

- Checkpoint `contact_conditioned_reference_bank_20260908_v1/campaign/segment_1200_1300_contact_v1/train/checkpoints/root_feedback_model_1300.pt`
  SHA `f91f8044a39aa8459e166d68897334012b5c11d1d77834c55f5352d2d47b5731`.
- Actor SHA `49825fd48b4a87061c0af37d6a8a164643cc350ca1afccbe8796e2783dd93e9b`;
  lineage `e61a73f1d1a85b2e32558b8fdeb572c5f4c1a6171b71178b6cd74954814c5a05`.
- Exported decoder SHA `faf4311dd610abdfcfbf1f8fa27be3f8b4817c3d5f467dd3865ad603fb1fc195`;
  measured CPU export parity maximum absolute error 1.132488251e-6, PASS;
  frozen encoder token parity exact.

Idle's full three-case replay is running in `policy1300_contact_evaluation`;
body's complete three cases follow. No learning-quality claim before comparison.
New comparison script independently recomputes lifecycle/root scores from hashed
state traces and refuses changes to source arrays, phases, physics, initial state,
runtime semantics or thresholds. It retains every landmark, every return metric,
and partial failures. Nine comparison tests PASS; actual policy1200 self-comparison
returns exactly zero changes for every metric. An improving mean cannot hide a
worse foot. Existing 149 curriculum/bank/trainer tests also PASS.

In parallel, a separately labelled full dance contact-reference experiment is
running at `contact_conditioned_reference_20260908_v6_dance/planned_full_dance`.
First attempt stopped before model construction on a transient Z:/WSL mesh-read
error. Both original and working mesh files remained nonempty and identical SHA
`3cd0d56fde14b73c1623304684805029971c4f84b596f9914e823ca70a107fd2`;
unchanged-model retry works. Ten bounded geometry steps accepted so far; maximum
ankle translation 19.499988 mm, ankle rotation 0.149500 rad, upper landmark change
31.856003 mm. Final physical collision/floor/support audit NOT yet passed.
This dance is NOT included in the two-source policy1300 training branch.

## 2026-09-08 Sydney, 00:14 UTC: corrected-source baseline complete; explicit contact training branch implemented

SIM/TRAINING ONLY. No robot, DDS, controller-mode, motor, deployment, commit or
push actions. Current policy1200 still FAILS tracking; no goal completion claim.

Both V5 contact-conditioned sources now have accepted complete lifecycle ramp
repairs. Idle entry/return COM changes are 0.975807/0.185079 mm; body return is
0.215279 mm. All source controls, generated ramps and 100 Hz ramp samples are
self-collision clear. These are geometric results, not working-policy claims.

Actual policy1200 comparison on these NEW references completed all six cases:

- Body: 1923 controls per case; root p95 nominal/X/Y
  0.046862647/0.048438996/0.046876226 m. Nominal ankle p95 left/right
  0.068901788/0.054153356 m; hands 0.061130678/0.083375650 m;
  head 0.029665167 m. Final standing joint maximum error 0.169271078 rad,
  root maximum error 0.045757450 m. Both feet still fail the unchanged 5 cm screen.
- Idle: 1491 controls per case; root p95 nominal/X/Y
  0.101052403/0.102243744/0.101151913 m. Nominal ankle p95 left/right
  0.105033163/0.143965655 m; hands 0.126488147/0.107610794 m;
  head 0.090955575 m. Final standing joint maximum error 0.159107827 rad,
  root maximum error 0.087990193 m. Tracking still fails.

No falls, midrun reset or fallback in any case. Small numerical improvements
versus earlier unconditioned references are NOT a policy improvement claim:
targets changed. These six full replays are the matched parent baseline for the
next learning comparison.

Added explicit `bounded_contact_conditioned_source_v1` curriculum and separate
contact-bank index. Original accepted raw bank bytes remain unchanged. The new
loader rederives both fixed offline registrations and independently checks all
serialized source arrays, FK, derivatives, root/arm/leg edit bounds, full rates,
100 Hz ankle/upper-landmark displacement, original-nearest stance points, floor
and physical self-collision. No optimizer status is accepted as a geometry proof.
Contact transition cannot crop, retime, change ownership/physics, expand the bank,
claim the old benchmark, hide remaining 8/27 support failures, or authorize hardware.

149 focused curriculum/registration/bank/trainer/geometry tests PASS; scoped lint
PASS. Independent actual-artifact checks pass all 1481 idle and 2345 body samples;
stored velocity channels also checked. Both maximum root-correction accelerations
are 0.497698784 m/s2, below the unchanged 0.5 bound. Raw original29 fidelity and
dynamic feasibility are explicitly NOT inherited by the corrected targets.

Prepared `C:/Users/camer/sonic23_sim_artifacts/contact_conditioned_reference_bank_20260908_v1/`.
Next segment is `campaign/segment_1200_1300_contact_v1`, same policy1200 parent,
actor/critic/optimizer state, encoder, nominal physics, reward and learning rate;
only reference profile changes. First launch stopped in preflight on a missing
50 Hz field in independent comparison; fixed before any samples ran. Relaunch
is validating the full six-case parent comparison before GPU training starts.

## 2026-09-08 Sydney,23:45UTC: whole-path contact correction passes geometry; actual policy comparison next

New contact-conditioned sources are explicit diagnostic variants, NOT silently
accepted original bank members. No additional PPO, robot operation or readiness
claim. The released planner target and actual29 controller trace remain distinct.

Rejected experiments retained:
`contact_conditioned_reference_20260907_v1` naive clipped IK produced excessive
foot drift and650 colliding100Hz poses; bounded least-squares fixed that numerical
defect but fixed-root V2 still had650 contacts/50 support failures. Coupled root+
legs V3 reduced this to109 contacts/26 support failures but exceeded2cm foot cap.
V4 whole-path initial affine step was infeasible. Diagnosis isolated contact
feature selection: flattened targets tie sole-corner heights and selected raised
heels instead of existing toes. Using nearest actual original sole sphere made
the same constrained geometry problem feasible; no physical/fidelity caps raised.

V5 solves all root translations, legs and arms together. Root attitudes and waist
yaw stay exact; root translation is bounded3cm/axis and added acceleration0.5m/s2
(tighter than old6m/s2). Legs may change0.2rad, arms0.3rad. Every original frame
retained, no retiming. Hard finite-grid correction caps:2cm ankle translation,
0.15rad ankle orientation,5cm hand/head displacement. All100Hz sole points must
clear floor, selected original stance point stays near floor, actual self-contact
model and joint/rate limits unchanged. Intermediate self-contact is elastic;
final geometry must pass independently after float32 serialization. Hand/head
landmarks use original TaskSpec offsets, including torso+0.35m head proxy.

Artifacts under `C:/Users/camer/sonic23_sim_artifacts/contact_conditioned_reference_20260907_v5_nearestsole/`:

- Idle:3 full accepted steps. Full741 controls/1481 half-grid samples have zero
  self-penetration, minimum sole gap0.099966mm, joint/rate/FK/correction checks PASS.
  Maximum ankle change11.287913mm, orientation0.075004762rad, upper landmark
  displacement13.369723mm. Nominal conditional support failures329->8/741;
  remaining389..395 and409; maximum solved torque ratio0.5787134234.
  Native motion SHA`9d9b2b296538a031d8eba94742551124593446da763d872b891d30477be1a23b`.
- Body-check:2 steps, full1173 controls/2345 half-grid samples collision-clear,
  minimum sole gap0.099970mm. Maximum ankle change3.159924mm; geometry/rates PASS.
  Support failures25->27, so body tracking failure is not explained chiefly by
  small sole placement errors. No support/dynamics qualification inferred.

12 focused contact-target/whole-path tests PASS, including analytical joint/root
and sole-height Jacobians versus independent finite differences, frozen attitude,
actual-nearest-contact selection, temporal bounds and no-op preservation.
Ruff passes for new contact implementation. Earlier35 ramp/terminal tests pass.

Fresh lifecycle preparation binds unchanged seven output channels plus original
named23 provenance; final offline calibration is explicitly after contact
conditioning (not an unchanged historical once-registration benchmark). Idle
source remains collision-clear; default generated ramps have58 contacts and
are being repaired using the already validated multi-joint ramp method.
Next: both full lifecycle geometry proofs, then actual policy1200 three-case
replays on these newly labelled sources. No training-bank acceptance inherited.

## 2026-09-08 Sydney,23:08UTC: dance lifecycle geometry fixed; support defects now measured

Continuing SIM-only. No hardware, commits, pushes or unchanged PPO continuation.
Policy1200 remains unqualified. No goal completion claimed.

The64 shoulder-only return candidates all passed joint/rate bounds but still
collided. Added bounded multi-joint arm-path optimization:10 actual arm joints,
only generated ramp interiors, maximum0.3rad beyond rate-corrected reference;
root/legs/source/standing remain exact. Physical1mm COM envelope, full rates
and independent100Hz collision gates unchanged. Solver removed all27 colliding
control poses in one accepted step; actual maximum arm change0.109909814rad,
return COM change0.000716983m. Serialized complete lifecycle and both203-sample
100Hz ramps have zero self-penetration. Source arrays remain bit-exact.
Artifact `C:/Users/camer/sonic23_sim_artifacts/once_registered_reference_20260907_v2/planned_full_dance/repaired_arm_path_v2/`;
lifecycle NPZ SHA`f0992db3d116b6f3a43adde310c14d3695035cdb4690e18859f55c1ae539d6b2`.
Loader independently verifies edit scope, recomputes rate correction and checks
FK/bounds/physical contacts/COM.35 scoped tests PASS; scoped lint PASS.
V17 independent original29 timestamp audit also passes all546 original frames.

Actual full policy1200 dance replay FINISHED in all three cases,1841 controls
each, no fallback/reset. Tracking FAILS: root p95 nominal/X/Y
0.419609/0.330289/0.395410m; nominal ankle p95 left/right0.543256/0.502080m.
Geometry acceptance is NOT a working dance or deployment readiness.

Independent unsmoothed50Hz inverse-dynamics necessary support diagnostics on
the unchanged nominal model/effort limits:

- Body-check:25/1173 frames cannot supply required support wrench; all other
  solutions within torque limits. Leg RMSE0.102530rad nominal, best lag0.
- Idle transition:329/741 fail, primarily369..401 and446..740. Raw source
  left foot slightly airborne while right foot penetrates floor up to~13mm.
  Original29 BONES source has the same mismatch: final left sole2.45..4.50mm
  above floor and right sole5.50..10.63mm below floor. This was inherited,
  not created by dropping six joints. Nominal leg RMSE0.124651rad, best lag-40ms.
- Full planned dance:1049/1091 fail;555 have no candidate ground contacts.
  The released planner's kinematic root/foot targets and uniform2x retiming
  are not a force-feasible motion certificate. Original29 controller trace
  remains distinct; it is not silently substituted for planned choreography.

Implementing explicit bounded contact-conditioned source diagnostic. It keeps
all frames, timing, root and upper-body channels; changes only leg reference
poses, with disclosed2cm ankle/0.15rad orientation/0.2rad joint correction caps.
Initial all-or-nothing flat-foot request exceeded scope at8 right-foot frames;
now correction is capped, not limits enlarged. Partial grounding and every
remaining force-support failure stay visible. No accepted training bank or
raw5mm original-fidelity claim will be inherited by this changed reference.

Next: complete contact-conditioned diagnostics, verify support and whole-path
geometry, then test actual policy behavior on valid references before training.

## 2026-09-08 Sydney,22:37UTC: calibrated1200 complete; feet remain the policy failure

Policy1100->1200 finished100 updates/819,200 new samples,8,601,600 cumulative.
Full actor/critic/optimizer/counters preserved at continuation;18 decoder tensors
and36,864 root-conditioner elements changed;10 encoder tensors unchanged.
Finite optimizer/critic and bounded std checks PASS; root max0.10158161073923111.
Checkpoint SHA`29c8234c83ef0a2068fb0ef172718ab8a8c421fefe1d495346d7498fdd2a333b`;
actor`6b0e7e565fdc41cbc1a614f2132ffc4ec51d1bad944a29e95a4a7abfe3b6a69a`;
lineage`e000dbec58c0cc66f24852bb7bbe2d690c82b667a7fb7ab4363166d233d4b8ee`.
ONNX decoder parity1.1920928955078125e-6 PASS; encoder exact.

All six full independent registered-source replays FINISHED; all tracking FAIL.
No postinitial state reset, shortened source or fallback. Same references and
generated ramp bytes as registered1100, not compared as unchanged-world against
older unregistered results:

- Body root p95 nominal/X/Y=0.047959/0.048470/0.047970m.
  Nominal feet0.073818/0.054078m, hands0.060367/0.083563m, head0.032084m.
  Both hands/head pass position thresholds in all three cases; feet do not.
  Final proof root max nominal0.045550m, joint error max0.163753rad.
- Idle root p95=0.109814/0.110273/0.109431m.
  Nominal feet0.117844/0.155916m, hands0.136156/0.095611m, head0.106603m.
  Final proof root max0.088766m, joint error max0.160799rad.

No unchanged next PPO block launched. Next policy diagnosis: original-reference
support conditions and unshifted full-source leg/foot phase error.

Dance V17 terminal refinement FINISHED PASS: original five full-grid gates AND
new terminal-braking-room gate all pass, zero physical self-penetration on1,091
controls/546 original timestamps. Source NPZ SHA
`ed4e54a0f722781be9171f839259b745af25901ff1acdfefa44d202b24081de8`.
Solver made3 accepted steps, then stalled on sub-micrometre preferred query
clearance; final independent physical collision/fidelity/braking gates passed.
Do not describe numerical preferred-clearance convergence as achieved.

With V17, independent ramp-only rate projection now succeeds: maximum joint
change0.027081574rad, max acceleration80.0000000004rad/s2 within unchanged
numerical audit tolerance, source/standing/root poses bit-exact. Return COM
change0.000411554m. Full source and timing unchanged.27 return control poses
still collide before shoulder repair. Added explicit optional composite rate+
shoulder repair and independent loader recomputation, retaining1mm COM and
full-control/100Hz-ramp collision gates. Composite repair currently RUNNING;
no accepted full dance lifecycle or policy replay claimed yet.
Six rate-projection tests and combined26 ramp/loader/terminal tests pass;
scoped lint passes. All1200 CPU replays completed BEFORE these loader/fitter
edits. Their historical code bindings remain historical; future continuation
must use fresh evidence, never silently repin an old report.

Artifacts remain under `C:/Users/camer/sonic23_sim_artifacts/`.
SIM/TRAINING only. No hardware, controller modes, commits or pushes.

## 2026-09-08 Sydney: full-dance geometry passes; calibrated training and terminal repair active

Full dance V16 FINISHED: all five independent control/original-time fidelity,
rate and physical self-collision gates PASS. All1,091 control poses and546
original timestamps are retained; zero penetrating frames on both grids.
Accepted native23 NPZ SHA
`db261dfd5180a1d611f40cb0b5bddd9d5e8a43d174004229549ba400e52d3063`.
Full excursion retained at2x duration; original-speed policy parity NOT proven.
22 accepted elastic steps plus extended bisection removed the remaining
clearance violation without relaxing original physical/fidelity limits.
V15 had failed with293 penetrating controls and93.760118mm maximum penetration.
Artifact: `C:/Users/camer/sonic23_sim_artifacts/collision_clearance_repair_20260907_v16/planned_full_dance/`.

Important separate failure: generated full-dance return ramp is NOT accepted.
The source terminal velocity is nonzero but the quintic return starts at rest.
Five joints violate80rad/s2 at the source/return join, worst148.329342rad/s2.
Right shoulder pitch specifically ends at-2.381161928rad, velocity-2.562654rad/s.
Even maximum allowed deceleration requires next position<=-2.400415009rad,
beyond unchanged lower bound-2.395448434rad. Thus ramp-only correction with
all source poses frozen cannot work; this is an explicit necessary bound, not
an optimizer timeout interpreted as global physical infeasibility.
The generated return also has27 colliding control poses. No dance policy replay
or hardware run claimed from the geometry pass.

Added independent terminal-braking constraint extension: four future maximum-
deceleration bounds per joint at50Hz, respecting existing5rad/s and80rad/s2.
Six deterministic envelope/Jacobian tests pass. The separate terminal-refinement
entry point keeps active training-bound files untouched, preserves all original
control/original-time task constraints, and adds terminal braking as a sixth gate.
V17 terminal refinement launched against acceptedV16, not a rejected warm start.
No terminal-fit success claimed yet. A ramp-rate projector now rejects an
impossible source endpoint before repeating a futile numerical projection.

Once-registered policy1100 complete three-case comparisons ALL still fail:
body root p95 nominal/X/Y=0.049398/0.050654/0.051550m; nominal source feet
0.074139/0.054723m, hands0.063383/0.108176m, head0.052557m.
Idle root p95=0.141785/0.144153/0.142342m; nominal source feet
0.108815/0.156287m, hands0.158578/0.093249m, head0.143663m.
Each includes all source frames, standing/entry/return, one policy, no fallback
or postinitial state resets. Calibration improves both versus unregistered1100
but does not establish dynamics/contact, held-out or teleop readiness.

Implemented explicit curriculum-only `once_only_source_start_se2_v1` with a
separately authorized reference transition. Original bank and source arrays stay
unchanged; registered world channels are independently recomputed, all joints,
height, timing, coverage and ownership retained. Both original sources freshly
reaudited with current collision implementation, including independent1,408/890
original timestamps. New bank has the EXACT old merged NPZ SHA
`53ea248c644a78ec36afa08c1e7f22aafb360eae8c1e92074c7315c870e490d0`.
The authoritative bank report is under
`C:/Users/camer/sonic23_sim_artifacts/once_registered_reference_bank_20260907_v1/bank/`.
98 curriculum/bank/registration tests and80 trainer/campaign/loader tests pass;
scoped lint passes. Source receipt checks rehash both historical originals and
byte-exact fresh audit copies; no historical implementation hashes repinned.
Policy1100->1200 full-state continuation is now RUNNING on both calibrated
references with unchanged foot objective, physics, limits, encoder and optimizer
settings. Six same-checkpoint complete registered CPU baselines bound before
training. New segment:
`.../once_registered_reference_bank_20260907_v1/campaign/segment_1100_1200_once_registered_v1/`.

No robot, DDS, motor, mode, arming, commits or pushes. Deployment remains false.

## 2026-09-08 Sydney: foot objective tested; one-time heading registration now isolated

At21:32UTC, policy1000 ->1100 finished100 updates/819,200 new samples
(7,782,400 cumulative). Actual full-state continuation and frozen-encoder checks
pass; no actor-only transfer, reset critic or optimizer. Checkpoint SHA
`db24a35e54181b1415448886b1dcf4306f50f11f67b3ca376a15ad8863c31dfa`, actor
`c030ac67b16579d7a4559809b564f95bb5da260acec152d6f7e7c958733e168d`, lineage
`0077e66c34ddbd105544dc4ff3c69c725db58ad9bf78c99b3306e9c283c23b79`.
All six independent full CPU replays complete; all remain tracking failures:

- Idle root p95 nominal/X/Y=0.164140/0.165638/0.166600m (worse than1000).
  Nominal source feet0.140472/0.165749m, hands0.156670/0.127272m,
  head0.243380m; final root max0.104619m.
- Body root p95=0.086503/0.082572/0.071215m (better than1000).
  Nominal source feet0.122260/0.091678m, hands0.116095/0.110768m,
  head0.116689m; final root max0.091094m.

This is a mixed result, not a validated policy fix. No further unchanged PPO
block launched. Source, repaired ramps and physics were identical to1000.
Artifacts: `.../two_collision_clear_reference_bank_20260907_v1/campaign/segment_1000_1100_feet_world_v4/`.
75 objective/environment/trainer tests pass; scoped lint passes.

Concrete startup mismatch found in trace and original C++ implementation:
body-check starts at-97.053804deg and idle at-85.361070deg, while the synthetic
standing reference starts at0deg. Thus the2-second kinematic entry imposes a
large turn, independent of source choreography. Original SONIC's
`UpdateHeadingState`/`ComputeApplyDeltaHeading` calibrates source heading against
initial robot heading. Added a separately labelled fixed SE(2) source-start
registration diagnostic, NEVER a per-step measured-state reanchor. Joint angles,
joint velocities, height, frame counts, timing and relative travel are preserved;
all world positions/orientations/velocities transform together. Ten tests pass,
including first-sample-only calibration, quaternion convention and invariants.
This is not the same world-frame benchmark and is not yet accepted training data.

Registered body source:all1,173 control poses remain exact-model FK consistent
and collision-clear. Generated return still needs the same+0.10/-0.15rad
shoulder bump; independently rechecked full1,934-pose and100-Hz ramp clearance
passes. New explicit pre-evaluation geometry mode avoids fabricating a policy
evaluation solely to fit generated ramps. No history resets, state edits, fallback
or target chasing permitted in the new replay. Policy1000's complete registered
body comparison is now running, under
`.../once_registered_reference_20260907_v1/body_check_001__A439/`.

Full dance V15 now tests explicit `elastic_squared_clearance_v3` from rejectedV14.
Only intermediate clearance rows get nonnegative optimization slacks; all original
linear root/joint/temporal bounds and every control/original-time task/hand/head
norm remain hard. Nonlinear line search must lower total squared clearance loss;
final independent physical collision/fidelity gates are unchanged. This removes
the old requirement that every existing penetrating pair improve/non-worsen in
each numerical step. Tests demonstrate opposed-contact escape and enforce hard
task constraints even when collision slacks remain. Combined75 collision/SOC/ramp
tests pass. V15 running; no accepted dance claimed.

Collision/refiner implementation edits occurred AFTER1100 training completed.
Old audit source-file bindings must not be silently repinned for later training;
any future bank consumption requires fresh independent evidence for changed code.
No hardware, DDS, motor, mode, commits or pushes.

## 2026-09-08 Sydney: policy1000 independently evaluated; horizontal drift remains

At21:07UTC, clean-reference policy900 ->1000 has finished100 updates and819,200
new samples (6,963,200 cumulative). Full-state continuation verification passes:
actor/critic/optimizer/counters preserved at900; all18 decoder tensors changed,
all10 encoder tensors frozen, finite optimizer/critic and exploration state.
Checkpoint SHA`bfe3353ad7bc08e49c4e552bc2290fba4021be3313b65f2416d15aed7e7ab859`;
actor SHA`9e6617b2a09e9229f0750bf2519454759be9549c32f7cc4bd6c0265fd84a3de7`.
ONNX max error1.430511474609375e-6 passes. All six complete independent CPU
lifecycle replays finish without reset/fallback. ALL still fail tracking.
Comparison uses identical accepted source bytes AND identical repaired ramps:

- Idle root p95 nominal/X/Y:900=0.182540/0.387094/0.434278m;
  1000=0.142428/0.144346/0.137983m. Nominal source feet0.122157/0.109258m;
  hands0.120695/0.114850m; head0.185883m. Final standing root max0.140010m.
- Body root p95 nominal/X/Y:900=0.113404/0.121906/0.109518m;
  1000=0.145847/0.124866/0.134341m (regression). Nominal source feet
  0.125544/0.164024m; hands0.101831/0.139357m; head0.100654m.
  Final standing root max0.149314m despite better joint posture.

Trace decomposition: source root Z absolute p95 is only5.0mm idle/22.5mm body;
horizontal displacement dominates root error. Body relative feet p95 also worsens
to97.6/144.9mm, so world drift is not the only error. Existing world-body reward
averages many links with0.3m scale. Implemented explicit optional
`root_and_upper_feet_world_v4`: add dense measured world-ankle squared distance,
mean over two feet, divided by0.05m squared, weight-1. Every existing reward,
physics/actuator/termination/observation/reference term stays unchanged. This is
an unproven training objective, not a fix claimed from reward or unit tests.
Next continuation requires full same-reference policy1000 baseline evidence.

V14 full dance using separately upper-bounded passing isolated pose guidance
has also finished REJECTED:6 accepted steps,299 penetrating controls, maximum
93.316076mm. Original fidelity and path gates pass, collision gate fails. No
rejected dance/walk data entered training. Next geometric investigation targets
the intermediate all-contact non-worsening constraint, not final safety gates.

Artifacts: `C:/Users/camer/sonic23_sim_artifacts/two_collision_clear_reference_bank_20260907_v1/campaign/segment_0900_1000_collision_clear_v1/`.
No hardware operations, commits or pushes. Deployment/generalization/contact
qualification remain false; neither clean poses nor survival imply dance readiness.

## 2026-09-08 Sydney: actual clean-reference PPO continuation running

At20:45UTC, policy900 -> 1000 continuation is actually training (67/100 updates,
548,864 new simulator samples at the last observed log). Same 128 environments,
64-step rollouts, PPO2epochs/4minibatches, LR5e-7, optimizer, actor/critic/counters,
source action codec, buffered timing, nominal physics and root/upper objective.
This is not an export or independent policy-tracking result yet.

Fresh independent reaudits of the accepted body/idle files pass every control,
original-time, temporal, physical-model FK and self-collision gate. Files are
copied byte-identically; old historical implementation hashes are not relabelled
as current evidence. Reaudit outputs:
`C:/Users/camer/sonic23_sim_artifacts/collision_clear_reference_reaudit_20260907_v1/`.
Separate BONES original-time audits again pass all1,408 body and890 idle timestamps.

Built two-clip/1,914-source-control collision-audited bank under
`C:/Users/camer/sonic23_sim_artifacts/two_collision_clear_reference_bank_20260907_v1/`.
Bank report SHA`0af3a596a3a1f8881ef5896548a5353e786681e8e667762dec493c2b7de1c023`.
The explicit repair plan preserves recording identities, full durations, source
counts and train splits, replaces only the two repaired references, and explicitly
quarantines unresolved dance/walk. It is NOT called an unchanged bank expansion.
No held-out/generalization/dance readiness follows from this smaller diagnostic bank.

Training curriculum independently reconstructs the SAME repaired entry/exit paths
as the six completed parent900 CPU evaluations. All six original source arrays
remain exact. Every selected source/lifecycle, prior checkpoint, old bank bytes,
quarantined identity, report and per-clip baseline is checked before constructing
the runner. Actual continuation contract records collision-clear generated phases,
old4 -> new2 scope, and actor/critic/optimizer/counter preservation. 34 bank repair
tests and71 curriculum/trainer/loader tests pass; scoped lint passes.
Training folder:
`.../two_collision_clear_reference_bank_20260907_v1/campaign/segment_0900_1000_collision_clear_v1/`.
Hardware implementation hashes remain7c94f882.../5c796525..., no robot operations.

V13 full fits have completed and remain rejected:

- Dance:12 accepted numerical steps, stop13;304 penetrating control poses,
  max96.649684mm. All original control/original-time fidelity and path gates pass.
- Walk:18 accepted steps, stop19;461 penetrating controls, max40.414712mm.
  All original control/original-time fidelity and path gates pass.

The new hand/head constraints prevent V12/V8 fidelity regressions but do not alone
solve full-path collision escape. Single dance pose1089 diagnostics with separate
10-cm hand/head bounds find6/8 independently passing solutions after adding a
0.995 optimization-only interior margin. Final original task/safe/root/1-mm query
and actual collision acceptance remain unchanged. V2 near-boundary solutions were
retained as rejected, not passed by loosening tolerance. V3 has real positive margin.
`C:/Users/camer/sonic23_sim_artifacts/collision_pose_escape_20260907_v3_upper_interior/`.
This is NOT temporal/full-path feasibility. V14 now uses the first passing V3 pose
as an explicit numerical objective for the complete dance path, with all V13
constraints still present. V14 running, not accepted. No physical infeasibility
certificate or arbitrary-29-DOF-dance claim is made.

## 2026-09-08 Sydney: generated ramp collisions repaired; full policy replays still fail

Simulation-only work continues on the requested goal. No robot/DDS/motor/mode
operations, no commit/push, and no further PPO on the old collision-invalid bank.

The two accepted source references below also needed generated entry/exit repairs:

- Body-check A439: 27 return-ramp poses overlapped, maximum 28.531 mm (hand/hand).
  Bounded endpoint-flat shoulder offsets (+0.10 left, -0.15 right radians at peak)
  clear all 1,934 full-reference poses and the generated ramps sampled at 100 Hz.
  Maximum ramp COM change 0.729 mm. Source, standing, root, legs and timing unchanged.
- Idle A026: 34 entry-ramp poses overlapped, maximum 7.676 mm (hip/right hand).
  Shoulder offsets +0.05/-0.10 radians clear all 1,502 poses and 100-Hz ramp samples.
  Maximum COM change 0.804 mm. Every original source channel remains bit-exact.

Artifacts are under
`C:/Users/camer/sonic23_sim_artifacts/collision_clear_ramps_20260907_v1/`.
Body lifecycle SHA `f291ecc503cf1b473cbcbac497fffc89ecc69166c03044a7efdedb5820da699f`;
idle lifecycle SHA `4e2dd3eed0fd7b8fccbfde03266c908158323e39cd6b8ed909720cba1f86346f`.
Existing safe joint/rate/acceleration limits and exact physical model unchanged.
These are sampled geometric checks, NOT continuous swept clearance or dynamics proof.

An explicit evaluator replacement loader independently rechecks the exact default
baseline, declared shoulder offsets, every source array, physical-model FK,
full-path bounds, full-grid collisions, and 100-Hz ramp collisions/COM before
inference. Fifteen loader/ramp tests and scoped lint pass. Both complete three-case
replays using mixed-bank policy900 have now finished without resets/fallback:

- Body root-position p95 nominal/X/Y: 0.11340444/0.12190592/0.10951837 m.
  Source-only tracking is identical to the previous test (only return changed).
  Nominal source feet p95 0.093046/0.137461 m; hands 0.102687/0.167005 m. FAIL.
- Idle root-position p95 nominal/X/Y: 0.18254043/0.38709402/0.43427819 m.
  Corrected entry makes tracking worse for this existing policy. Nominal source
  feet p95 0.162413/0.164299 m; hands 0.179584/0.188977 m. FAIL.

No new dynamic or deployment qualification. Geometry repair alone is not a policy fix.

Dance V10/V11 stalled; V12's explicit sum-squared collision merit accepted four
steps but still left 300 penetrating control poses, max 99.749 mm, and failed
right-hand p95. Walking V8 cleared contacts but failed left-hand p95 0.114106 m.
Neither is an accepted training reference. The earlier 5/8 passing isolated dance
poses enforced the original six per-frame norms, NOT separate hand/head 10-cm
position envelopes; they do not prove complete-path feasibility.

Added explicit `--protect-upper-landmark-p95`: fixed per-frame position budgets
on both hands/head on every control AND every original timestamp. Protected rank
identities are selected once from the accepted parent, not the rejected iterate.
Both order statistics used by NumPy's linear p95 remain below the old 10-cm gate;
remaining outliers retain bounded parent-reference errors. This is a stricter
fitting surrogate, not a relaxed final gate. Original six hard task norms remain.
78 targeted upper/collision/interpolation/guidance tests and scoped lint pass.
V13 full dance and walk fits are running with these constraints, not yet accepted.
New constraints already reject steps that would repeat the hand-tracking failure.

Curriculum support for independently reverified generated ramp references and an
explicit repaired-bank/quarantine transition is being implemented. No new training
has started. Temporarily omitted references must remain explicitly unqualified;
training a small clean subset cannot establish arbitrary-dance or held-out parity.

## 2026-09-08 Sydney: two full collision-clear references; deployment still unqualified

At19:32UTC, simulation-only work continues. No robot/DDS/motor/mode operations,
no commit/push, no further PPO on the unchanged interpenetrating reference bank.

Accepted full reference repairs (geometric, NOT dynamic qualification):

- Body-checkA439 V7:zero self-penetration at ALL1,173 control and1,408 original
  timestamps. All original fidelity/COM/feet, safe joint, root, velocity and
  acceleration/serialization gates pass. Three warm-start numerical steps.
  Motion SHA`a1f4abb7e0b44f95e45b88e5b0abe269369a0a8be10e2da3fafdede35125e3d1`.
  Folder`C:/Users/camer/sonic23_sim_artifacts/collision_clearance_repair_20260907_v7/body_check_001__A439/`.
- IdleA026 V9:zero self-penetration at ALL741 control and890 original timestamps;
  every original gate passes. V8 first cleared collisions but failed6 original
  right-foot positions and2 orientations. V9 constrains those original-time task
  norms DURING fitting and restores all gates in ONE numerical step, no tolerances
  relaxed. Motion SHA`97acd79af697347507f9873e799acbd8643d862f90abc8821fb7d239db948b8c`.
  Folder`C:/Users/camer/sonic23_sim_artifacts/collision_clearance_repair_20260907_v9/idle_right_to_idle_R_002__A026/`.

The new original-time Jacobian propagates linear joints/world translation and
SLERP root rotations through both neighboring control knots. Finite differences
check all87 input coordinates at zero/small/nonzero rotations. Existing direct
native baseline is independently recomputed from original29 FK, not replaced by
an easier reference. Every original task norm joins the unchanged control-grid
norms; all final original-time and serialized physical-model checks still rerun.
Unique ancestor diagnostic seeds may be reused only from already-bound parent
provenance; missing/ambiguous seeds fail. Rejected originals may warm-start ONLY
explicit fully covered original-task restoration, not ordinary acceptance.

All9 new complete CPU lifecycle evaluations finish without resets/fallback;
ALL remain tracking failures. Body-check has1,923 controls per case; idle1,491.
Root-position p95 nominal/X/Y(m):

- Repaired body, policy800:0.09613096/0.06855281/0.09092708.
- Repaired body, mixed-bank900:0.11340444/0.12190592/0.10951837.
- Repaired idle, mixed-bank900:0.14866790/0.09886037/0.07560593.

Repaired body900 nominal feet p950.093046/0.137461m exceed5cm; left/right hands
0.102687/0.167005m, head0.092831m. Geometry repair alone is NOT a policy fix.
No root-state hardware estimator, dynamic contact, safe handover or teleop
qualification claimed. No continuous-between-sample collision proof claimed.

Other retained rejects:

- Dance V7:309/1,091 control and153/546 original poses penetrate, max94.719mm
  at control1089. All old control/original fidelity and path bounds pass.
  Explicit minimum-change objective stalled after10 accepted steps at11.
- WalkingA052 V8:17 steps clear all control/original query collisions, but
  left-hand p95 exceeds unchanged10cm limit. Final control fidelity fails;
  original-time fidelity therefore not run. Do not train this candidate.

Bounded isolated-pose diagnosis on dance1089 finds5/8 alternate starts that
clear collisions and satisfy every original per-pose norm and safe bound.
Largest joint change~0.4808rad for first passing seed. This is NOT a full-path
solution or impossibility certificate. Root-L1 constraint rewritten as its exact
8 linear facets and task norms as equivalent smooth squared norms after initial
finite-difference checks exposed cusps; final Jacobian error6.2401e-5.
Report`C:/Users/camer/sonic23_sim_artifacts/collision_pose_escape_20260907_v1/report.json`.

Dance V10 now tests the first independently rechecked passing isolated pose as
a LOCAL NUMERICAL OBJECTIVE target over100-control radius, not replacement
choreography or a state rewrite. All full-path physical/temporal/task gates and
both timestamp grids stay enforced. Warm start is rejected V7, still rejected.
Output`C:/Users/camer/sonic23_sim_artifacts/collision_clearance_repair_20260907_v10/planned_full_dance/`.

New bank builder/campaign gates reject legacy FK-only collision-unsafe banks.
New bank metadata binds accepted motion SHA and physical-model collision audits
on both full grids. Campaign also rechecks every generated lifecycle control pose,
including acquisition/return ramps. Historical audits remain historical: new
bank assembly still needs fresh fully bound audits after implementation changes.
68 current solver/interpolation/guidance tests plus51 bank tests pass; scoped
lint passes. These tests are code checks, not robot or policy qualification.

## 2026-09-08 Sydney: between-knot collision constraints and minimum-change repair

At18:59UTC, simulation-only work continues. No robot/DDS/motor/mode or git
commit/push actions. V6 completed, neither reference accepted:

- Dance:18 accepted numerical steps, stopped at19;308/1,091 control poses still
  penetrate, max96.246mm. One left-foot orientation gate fails at control628;
  original-time fidelity therefore not run. Candidate retained as rejected only.
- Body-checkA439:all1,173 control poses collision-free, all old control and1,408
  original-time fidelity gates and serialized path bounds pass. Original-time
  collision gate alone fails:11 hand–hand overlaps, max8.077mm at original955.
  This exposes collision between clear control knots; not a pass or training source.

Implemented explicit `minimum_change_v2` engineering objective: minimize weighted
movement from CURRENT iterate instead of also improving unrelated task errors.
All existing hard task norms, safe joint envelope, root, speed, acceleration and
independent numerical acceptance thresholds remain unchanged. Objective change
is explicit in reports; no claim of equivalence to the previous LSQ objective.
Added exact collision interpolation at the union of EVERY control and original
timestamp; sparse Jacobians propagate through both adjacent control knots.
Finite-grid only: continuous swept collision/dynamic feasibility NOT proven.

V7 body-check reuses hash-bound rejected V6 poses with identical parent/source,
models and old full-path gates. All acceptance tests rerun; rejected artifact is
not promoted. Three full numerical steps clear the union-query margin with zero
protected-task excess; final independent serialized/original checks still running.
V7 dance starts from its original complete source, same new objective/grid;
three full steps reduce query overlap109.282→95.435mm with zero task excess.
Both output under `C:/Users/camer/sonic23_sim_artifacts/collision_clearance_repair_20260907_v7/`.
66 targeted solver/collision/sampling/provenance tests and scoped lint pass.
Do not resume PPO on the unchanged interpenetrating four-reference bank.

## 2026-09-08 Sydney: collision repair widened numerically; no physical-limit changes

At18:37UTC, simulation-only repair remains active. No robot/DDS/mode/motor or
git commit/push actions. Hardware implementation hashes rechecked unchanged:
gantry CPP`7c94f882ad4e7faccd8ba1de6571167fbf82f2c9fc12eaa6f9b7f0c00a017ddf`,
core HPP`5c7965251b49d4f3005e9802aebddb03f60534603992c11837078e5ac41badaf`.

Retained collision-repair trials, ALL rejected, no use as training references:

- V1:5 accepted numerical steps, stopped at6. Dance penetrating frames309→259,
  max109.282→100.039mm. All original control, source-time and temporal gates pass;
  both new collision gates fail (131/546 original samples penetrate).
- V2:worst-contact-first restoration,7 accepted steps, stopped at8. Max98.709mm,
  still309/1,091 penetrating. All old gates pass; collision gates fail.
- Diagnostic exposes old `maximum_joint_change_rad=0.6` as an artificial search
  neighborhood, not a motor rating: at frame1088 right shoulder pitch reaches
  neighborhood lower-1.88351rad, while unchanged safe lower is-2.39545rad.
  Existing physical/safe envelope, root correction,5rad/s,80rad/s², torque and
  source-fidelity gates are NOT widened. Explicit new
  `full_existing_safe_envelope_v2` removes only that seed-neighborhood cap.
- V3 widened search:original-row numerical audit rejects1.92e-8/1.22e-8 residuals
  versus unchanged1e-8 check. V4 tighter1e-11 solver precision returns AlmostSolved.
- V5 permits a reduced-accuracy numerical iterate ONLY after independent original
  row/norm verification, with no optimality claim. Still rejects1.23e-8/1.22e-8
  original-row errors. V3–V5 accept zero steps; original path retained unchanged.

Implemented analytically eliminated residual SOC formulation, exact same
objective and constraints: minimize0.5||r+Jδ||² plus the same posture cost,
subject to unchanged affine norm and linear bounds. Dance variables74,188→
31,639 by eliminating42,549 auxiliary residual variables/equalities. Regression
tests compare this against the previous augmented form at two objective scales;
also reject infeasible contact constraints.33 affine/collision tests pass;
38 prior protected/precision/AlmostSolved tests pass. Scoped lint passes.
Legacy solver defaults preserve their previous behavior. Independent original
row/norm tolerance remains1e-8; final nonlinear/serialized gates unchanged.

V6 now runs same full-safe search with eliminated residual formulation on two
complete, preselected train/local sources. NO code edits to bound helpers while
either job runs. Dance at iteration14:109.282→100.563mm maximum overlap, globally
small line-search steps0.125; intermediate protected normalized residual~0.00054
is NOT accepted motion evidence. Body-checkA439 at iteration11:76.518→29.730mm,
full steps1.0, protected normalized residual0. Both bounded at48 iterations and
must rerun original source times, serialization, and collision checks afterward.
Outputs: `C:/Users/camer/sonic23_sim_artifacts/collision_clearance_repair_20260907_v6/`
with separate `planned_full_dance` and `body_check_001__A439` folders.

Reference pose1088 rendered and visually inspected: right hand intersects head.
This is a STATIC REFERENCE, not a policy rollout or robot-camera view. Render uses
unchanged physical model/no integration and its existing640×480 framebuffer;
initial960px request failed before output, then corrected without model edits.
Image: `C:/Users/camer/sonic23_sim_artifacts/four_clip_reference_bank_20260907_v1/dance_reference_frame1088.png`.
Separate original29 geometric check on bound original29.mjb: planned546/546 and
recorded546/546 have ZERO self-penetration. Self-collision filtering is enabled
(disableflags0,38 active geoms including torso/arms/wrists/ground). This is a
model-specific geometry comparison, not a matched dynamical parity certificate.

Mixed-bank900 additional measurements (all still failed qualifications):
108 full reference timelines/702 suffixes/940 resets. Actor SHA
`6368af0f8f6548106df97473eda8ecd926755d9f0b8ca767b82648d822bff207`.
Dance leg RMSE nominal/X/Y0.18283/0.17869/0.18089rad, lag40ms, all-joint RMSE
0.20648/0.20394/0.20464rad. Original29 head p950.36549/0.42027/0.55445m;
feet nominal0.39078/0.46332m, X0.38386/0.46883m, Y0.51611/0.62114m.
Walking leg RMSE0.19084/0.19179/0.19409rad; lag20/40/40ms. No accepted metrics
were phase-shifted. Standing lag diagnostic hits400ms search boundary, not a
reliable timing correction. Do not promote this bank900 or restart unchanged PPO.

## 2026-09-08 Sydney: self-collision defect confirmed; full-path geometric repair running

Goal unfinished. Simulation/training only; no robot, DDS, motor, mode, commit or
push actions. Stop further unchanged training on the current four-clip bank.
Its existing accepted FK/control/original-time checks did NOT include self-
collision clearance. That acceptance is not physical-reference feasibility.

The mixed-bank800→900 branch completed819,200 new samples, cumulative6,144,000
on this branch. Full actor/critic/optimizer/counter continuation checks pass;
all10 encoder tensors unchanged,18 decoder tensors and36,864 root weights
changed. Checkpoint SHA`7c5fca71cd812003048b1ef9ca5db06f6949315cd50b74c867bcb9ff926c9484`,
lineage`2fb45a30bbb7fb21d5671b9cb6e1f17525435e9cc65cbabbef44b51ee017ffb6`.
Root weight max0.06150853; ONNX parity max error1.1920929e-6. This mixed-bank900
is distinct from the rejected world-priority900. All12 complete CPU lifecycle
cases finish without reset/fallback; ALL still fail tracking qualification.

| Source | Controls per case | Root p95 nominal/X/Y,bank800→bank900(m) |
| --- | --- | --- |
| Full planned dance | 1,841 | 0.37155/0.38140/0.34859→0.34269/0.35406/0.48809 |
| Idle transitionA026 | 1,491 | 0.13974/0.13513/0.12999→0.08786/0.08860/0.35578 |
| Body-check standingA439 | 1,923 | 0.08446/0.08470/0.08675→0.15562/0.15209/0.14490 |
| Hands-back walkingA052 | 1,219 | 1.91611/2.04465/2.03970→1.64846/1.68162/1.67229 |

New complete pose audit uses the exact unchanged CPU dynamics model, compiled
SHA`80c82f5374bed69423580b4d2085e2103bfd83b8691dfe1f7699d2434e1561a8`.
Every source native23 joint name/order and ALL derived body poses were checked:
max FK position mismatch≤4.78e-7m; no joint-order/model mismatch explains this.
Scratch `mj_forward`, zero integration; exclude world/ground contacts and positive
margin contacts. Geometry overlap is NOT the training sensor's10-N force-history
reward. No force claim is made by this geometric diagnostic.

| Reference | Penetrating/control frames | Max overlap | Worst pair/frame |
| --- | --- | --- | --- |
| Full planned dance | 309/1,091 | 109.282mm | head–right rubber hand/1088 |
| IdleA026 | 391/741 | 20.373mm | right hip-yaw–right rubber hand/326 |
| Body-checkA439 | 913/1,173 | 76.518mm | left shoulder-yaw–right rubber hand/522 |
| WalkingA052 | 469/469 | 68.675mm | left–right rubber hands/451 |

Walking additionally intersects both arms with torso; dance also intersects
left hand/head. Measured800 nominal poses instead show6/255/822/469 penetrating
frames with max4.463/1.129/1.535/2.113mm. These conflicts are a concrete reference
defect, not proof that they explain all drift or that the dances are impossible.
Evidence: `C:/Users/camer/sonic23_sim_artifacts/four_clip_reference_bank_20260907_v1/policy800_self_contact_geometry_v2.json`.
V1 retained. V2 additionally binds physics config/compiled model, validates FK,
labels body pairs/worst frames, and freezes source closure before work. Its first
attempt rejected extra provenance keys at the strict library-schema adapter;
fixed by explicitly passing the seven motion channels, not weakening validation.

Implemented offline collision-aware whole-path SQP using existing protected-task
SOC constraints and unchanged original root/joint/rate bounds. Collision query
uses a separate model copy with30mm query horizon; physical model untouched.
Signed-distance Jacobians use exact contact normals and point Jacobians. Actual
worst-pose finite-difference gradient max errors:8.74e-7/3.21e-5/3.12e-5/6.34e-6.
Numerical intermediate clearance restoration is never an accepted reference.
Final acceptance additionally requires zero robot-robot penetration at every
serialized control and every original timestamp, plus ALL old FK/path gates.
No collision penalty removal, dropped frames, target replacement or gate easing.
27 focused contact/repair tests pass;71 combined contact/objective/launcher tests
pass. New full-dance repair is now running with48 bounded numerical iterations,
all1,091 controls/546 original samples. First step reduces max overlap109.282→
104.301mm while protected task constraints remain satisfied. No final repair
pass yet. Output if completed:
`C:/Users/camer/sonic23_sim_artifacts/collision_clearance_repair_20260907_v1/planned_full_dance/`.

After all bank900 training/export/evaluation jobs finished, fixed FUTURE top-level
reward metadata to take the actual selected objective's root weight. Added4
profile tests. Old world-priority900 artifacts stay unchanged and retain their
documented top-level-10/nested-and-actual-30 metadata inconsistency.

## 2026-09-08 Sydney: dance-only world-priority branch stopped; four-motion continuation validating

Goal unfinished; simulation only. No robot/DDS/mode/motor/commit/push actions.
The800→900 world-priority trial completed819,200 additional samples, cumulative
6,144,000 on that branch.96 full timelines/616 sampled suffixes/840 resets;
full-state continuation, finite optimizer/critic and export checks pass. Encoder
unchanged, decoder18/all36,864 root weights changed; root max0.07737306.
Checkpoint SHA`989bc10c5c0fae16b3d73892546c83e2ebc8cdb21dc91c90af9100ce9b7150b1`,
actor`6220f2534c62f06b8aa53be57acfd2951ef65ca11d8e1383332dd10faa1f6416`,
lineage`19eaa6400aeedcb898492f4669ebd0b0c8cda383331316342727290b8488983b`.
Decoder export max error7.1525574e-7. This is the world-priority900 branch,
NOT the forthcoming mixed-data900 branch; iteration numbers alone are ambiguous.

All3×1,841 full-dance controls complete, no reset/fallback, all tracking screens
FAIL. Root p95 nominal/X/Y0.50926/0.37888/0.31889m versus8000.37155/0.38140/
0.34859m. Leg RMSE0.17115/0.17048/0.17098rad, lag40ms. Nominal source tracking
regresses strongly, two push cases improve slightly, mean root p95 worsens for
the second consecutive full-reference block. Stop this dance-only tuning branch;
do NOT run an unchanged900→1000 world-priority block. Final standing joint errors
0.29760/0.23959/0.23526rad worsen, while standing root errors0.18609/0.07028/
0.04331m improve. Original29 head p950.51912/0.43643/0.38989m; original feet
nominal0.55254/0.58711m, X0.38162/0.52998m, Y0.37999/0.42829m.
Source projection loss0.0494552/0.0460833/0.0509646rad². Walking900 also completes
all3×1,219 but fails path tracking: root p951.97796/1.98199/1.89909m.

Known metadata defect found after this rejected trial: actual constructed v3
reward and its nested objective contract both use-30, but the legacy top-level
`root_world_tracking_error_weight` metadata field still says-10. Do not silently
rewrite old artifacts or call the trial fully metadata-consistent. Fix future
contract generation after the active v2 bank job; v2's actual/declared weight-10
already matches. This does not remove or improve the measured failed outcomes.

Measured800 full-dance video rendered from all1,842 stored qpos states/50Hz,
36.82s integrated, no new policy/dynamics run. Video SHA
`cbcc38d71e033d5bf96f9f95f5b7b00b50675aafc3572a9c26e18aaad1aabeff`,
trace`adfb3c455f32e4c7cd9ba3d3e8e44bdfb46e3cf2233adeea371bbd2ec3cd6240`.
Path: full-range `segment_0700_0800_upper_posture_v2/evaluate_updated/nominal.measured-simulation.mp4`.
App opening is queued, not confirmed played.18s frame inspected. New reusable
`render_g1_true23_measured_trace.py` checks trace/report/model hashes and refuses
overwrite; the rendered motion is not a qualification certificate.

All FOUR checkpoint800 baselines are now complete: dance3×1,841, transition
A0263×1,491, standingA4393×1,923, walkingA0523×1,219. Transition root p95
0.13974/0.13513/0.12999m; standing0.08446/0.08470/0.08675m; the walking~2m
failure remains explicit. The candidate comparison index binds each report and
the exact four-reference bank. No held-out-generalization claim.

Implemented explicit `--reference-bank --allow-reference-bank-transition` for
local evaluated continuation: preserve every previous source AND lifecycle
array, admit only the exact bank, validate all three scheduled outcomes for
EVERY clip against one parent checkpoint, and re-feed all saved received-only
encoder/root/timestamp inputs. Cross-clip dynamics model/gains/limits/config
must match; FK model and dynamics compiled identities are intentionally not
equated. Default single-reference guard stays enforced.102 focused tests pass;
then52 focused launcher/bank tests pass after fixing the remaining one-reference
launcher check. First launch stopped before outputs/training at that leftover
check; both output directories were absent, so corrected retry overwrote nothing.

The new mixed-data800→900 branch keeps800's v2 objectives, full actor/critic/
optimizer/counters, model/timing/action codec and128×64×100 batching. Only bank
membership expands from one to four. At17:34UTC it is still in preflight CPU
validation, NOT yet GPU training (PID489 active,~4.25GB RSS). No further source
edits while the bound run is active. Output root:
`C:/Users/camer/sonic23_sim_artifacts/four_clip_reference_bank_20260907_v1/campaign/segment_0800_0900_multimotion_v1/`.
After training, export and compare every clip again before any further block.

## 2026-09-08 Sydney: full-range800 evaluated; world-priority900 running; walking repair passes

Simulation only. Goal unfinished: no dance fidelity, generalization or live-teleop
qualification. No robot, DDS, mode, motor, commit or push commands. Historical
status below is a time-stamped log, not the current deployment state.

The700 old-reference block completed and passed full continuation/export checks.
On the NEW100%-excursion reference,700 completed all3×1,841 controls, no reset or
fallback. Its final standing posture improved versus500, while world tracking
was mixed. The failed600 full-range X case remains retained.700 was selected for
explicit reference adaptation, not declared universally better than500.

Integrated the same-source full-excursion transition contract after700 finished:
old/new reference reports, unchanged physical models/timing/limits/frame counts,
and the passed complete original-timestamp audit are bound into new lineage.
Reference-array changes remain rejected by default.63 focused transition/parser/
campaign/continuation tests passed. This is NOT a same-reference resume claim.

700→800 trained819,200 additional samples on the full-range reference, cumulative
5,324,800 critic samples.95 full reference timelines and644 sampled suffixes;
869 resets. Full actor/critic/optimizer/counter continuation passed; encoder10
tensors unchanged, decoder18 and all36,864 root weights changed. Root weight
max0.0612598732; finite critic/optimizer. Checkpoint800 SHA
`3c60b849846322cb0a80182b5237ca22d9d3b0d0bb4684e678abf4248c6dda52`,
actor`b1f328ae49d46899c3874a18c02f171ddee10ac5eac0df680282e44a7a24115f`,
lineage`4a71c468114c998301fea8386f0f74cbdf261dea760a108a223f0e7f5d3e89a7`.
Decoder export max error1.0728836e-6. Evidence:
`C:/Users/camer/sonic23_sim_artifacts/planned_dance_full_excursion_20260907_v1/campaign/segment_0700_0800_upper_posture_v2/`.

| Same full-reference CPU case | Root p95,700→800(m) | Leg RMSE,700→800(rad) | Final standing max joint/root800(rad/m) |
| --- | --- | --- | --- |
| Nominal | 0.33179→0.37155 | 0.16151→0.16886 | 0.19686 / 0.23068 |
| Standing push X | 0.26278→0.38140 | 0.16235→0.17150 | 0.18145 / 0.22469 |
| Standing push Y | 0.43352→0.34859 | 0.16048→0.17266 | 0.22755 / 0.25530 |

All3×1,841 finish; all landmark screens FAIL. Lag remains40ms, no metric shift.
Standing joint posture improves across all3, but standing world-position error
worsens across all3. Original29 world head p95 at8000.43251/0.45414/0.39396m;
feet nominal0.38689/0.50254m, X0.40408/0.53185m, Y0.37181/0.49094m.
Source projection loss0.0488581/0.0504797/0.0591000rad²; projected means
49.86%/51.70%/53.99%. These are measured failures, not qualification.

Training event logs show last reported episode arm penalty-2.3840 versus
root-tracking-0.4412 (iteration792). This supports, but does not prove, a reward
tradeoff. Implemented explicit `root_and_upper_world_priority_v3`: ONLY existing
root-world tracking weight-10→-30; all other reward functions/weights/params,
actions, observations, events and terminations unchanged.91 focused tests pass,
including exact constructed-config comparison; scoped lint passes.800→900 now
running at128×64×100, unchanged encoder/model/action codec/timing/reference/PPO
and optimizer state. Evaluate all3 full cases before any further training choice;
do not repeat through two worsening world-tracking evaluations.

BONES-SEED A439 body-check unseen TRAIN-split probes500 and700 both complete
all3×1,923; all landmark screens FAIL. Root p95 improves0.11687→0.05469,
0.12622→0.08252,0.11570→0.05977m; standing joint error improves, but leg RMSE
worsens~17%. This is neither held-out validation nor universal-dance evidence.

Walking `walk_hands_on_back_180_start_R_001__A052`:563 original120Hz samples,
469 control50Hz samples,1.9985765x duration,100% excursion. Initial control-grid
fit passed but original timestamps failed COM1/left-foot orientation5/right11.
Added explicit bounded grouped original/control-grid repair: seven separated
groups,308 selected joint coordinates, at most128 variables per group and0.01rad
correction, root/unselected pose values unchanged. All original acceptance gates
and full-trajectory5rad/s,80rad/s² bounds remain unchanged.33 focused tests pass.
v1 fixed foot defects but retained one serialized COM failure; REJECTED artifact
kept. v2 fits an additional1e-7m inside the same COM limit, not a relaxed gate.
Largest actual correction0.003831774rad. Full control and all563 original-time
checks PASS; separate independent audit also PASS, failures[]. Accepted NPZ SHA
`a790d344fe09835ef1e3f6a47851a0fbd9ad0fdf2c2e82fd8df3a5dc04752a33`.
Evidence: `C:/Users/camer/sonic23_sim_artifacts/bones_seed_grouped_repair_20260907_v2/walk_hands_on_back_180_start_R_001__A052/`.
No source samples, root, tempo or excursion changed by repair; no optimizer
optimality, contact or dynamics claim. Policy700/800 walking probes now both
complete all3×1,219 controls without reset/fallback, but all landmark screens
FAIL. Root p95 nominal/X/Y7002.10059/2.03123/2.03844m versus8001.91611/2.04465/
2.03970m. Leg RMSE7000.20636/0.21014/0.20707rad versus8000.20319/0.20672/
0.20744rad; all lag20ms. Survival is not path tracking. These are unseen train
data, not held-out data, and expose poor locomotion transfer from dance tuning.

Built a four-reference local simulation bank without changing any member's six
training arrays: planned dance1,091, transitionA026741, standingA4391,173 and
walkingA052469 frames,3,474 total at50Hz (~69s). Every member's accepted control
fit and complete original-time audit is checked and hash-bound. BONES rows must
exactly match the frozen full metadata index and original capture-group train
split; the local planner remains explicitly local_regression, not held-out.
No resampling, root alignment, joint deletion, blended transitions or file
publication.21 synthetic builder tests pass; scoped lint passes. The actual
serialized bank's frame slices match every member bit-for-bit.
Bank SHA`6fceba22da8b19d09a14f30e6d4096b4e67f090cb9ff761a1d5760e566ae79cf`.
Evidence: `C:/Users/camer/sonic23_sim_artifacts/four_clip_reference_bank_20260907_v1/bank/`.
This is a small local bank, not broad corpus/held-out/dynamics qualification.
Not yet used in training: multi-reference continuation still requires explicit
integration and separate complete CPU evaluation of every selected clip.

## 2026-09-08 Sydney: full-excursion reference accepted and independently audited

Concrete reference correction: all546 original planned29 frames now have an
accepted1,091-control native23 fit at2x duration and100% excursion. The previous
10% travel/amplitude reduction is gone. Existing joint/root/temporal/protected
task limits unchanged. All control frames pass, no protected failures; newly
recomputed FK at all546 original timestamps also PASS, failures[]. No reference
or measured path reanchoring. The bounded intermediate feasibility-restoration
method succeeded where the first strict linearized subproblem rejected.
Reference SHA`cee2ac43f35aeba251da621f89f0b4a5b06284413a93c16bae2b12af0d56c0ba`.
Same original named29 SHA`c8ea3b67e726a33b0e74d5ee67ae5e7698d8886d949226fe0436f764279d25dd`.
Reference-to-original p95 feet1.85/1.91mm, hands7.05/7.35cm, head4.65cm;
maximum foot error4.22/4.95mm. This remains2x-duration, kinematic acceptance,
not original-speed, continuous-time, contact, dynamics, policy or teleop parity.
Evidence: `C:/Users/camer/sonic23_sim_artifacts/planned_dance_full_excursion_20260907_v1/restored_protected/`.

New read-only `audit_g1_true23_planned_source_timestamps.py` reconstructs the
named input from `planned_qpos50`, rejects recorded policy poses/changed hashes,
and reuses the unchanged independent original-timestamp FK gate.29 tests pass
with the common audit suite; scoped lint passes. Failed weighted/strict artifacts
remain retained beside the accepted reference; none were relabelled.

Policy500 evaluated unchanged on this full-range reference: all3×1,841 controls
complete without reset/fallback; all landmark screens still FAIL. Root p95
nominal/X/Y0.33167/0.35616/0.29253m. Final standing max joint errors
0.36567/0.35200/0.37471rad, max root errors0.14215/0.34250/0.28342m.
Original-source head p95 now0.35639/0.40175/0.32653m, versus the SAME policy's
0.81832/1.04674/0.86380m on shrunkenv5. This improvement is due to the reference
correction, not additional training or a different comparison denominator.
Original-source feet p95 nominal0.40167/0.42921m, X0.36627/0.41941m,
Y0.33017/0.28578m; substantial controller tracking error remains.

Policy600 full-range nominal/Y complete, but X stops at1,042/1,841 for absolute
height/tilt. Its incomplete case is retained, not cropped into a passing metric.
Policy500 is the stronger full-range starting candidate so far. The ongoing
600→700 experiment still uses shrunkenv5, untouched mid-run. A new explicit
same-source reference-transition validator has20 passing tests, rejecting
changed models/timing/limits, cropped frames, other recordings, and failed
original-time audits. Not integrated into the active trainer until700 finishes.
Added a separate hash-bound full-range `source.spans.json`; no training launched
on the new reference yet. BONES-SEED body_check001A439 policy500 evaluation is
running separately; it is an unseen train-split probe, not held-out qualification.

## 2026-09-08 Sydney: buffered600 improves posture, worsens root; bounded700 active

Simulation only; goal unfinished. No robot/DDS/mode/motor/commit/push actions.
The500→600 upper-posture experiment completed819,200 transitions, cumulative
3,686,400 critic samples;96 full reference timelines,616 sampled suffixes,
128 standing/712 source resets. Full actor/critic/optimizer/counter continuation
checks pass. Encoder10 tensors unchanged, decoder18 changed, all36,864 root
weights changed (maximum absolute0.04537977); all critic/optimizer states finite.
Checkpoint600 SHA`7217514cf6ed7454b1c025cd5f52f4e8c7548b24759a327a9f35c82e598f24d1`,
actor`bf770c5762f9f5a5a8a7fb07a8a949eb41c0d24d9297004fcb9ff67642137f08`,
lineage`07544b4af8b13d8c26bfdf9ed3b956d99139dd8314e075922eff0e6fd97b9096`.
Export parity: decoder8.34465e-7, encoder0. Encoder ONNX file hashes include
export metadata; frozen tensor identity is the unchanged encoder-state hash.
Evidence: `C:/Users/camer/sonic23_sim_artifacts/received_source_horizon_20260907_v1/campaign/segment_0500_0600_upper_posture_v2/`.

| Full CPU case | Root p95,500→600(m) | All-joint RMSE,500→600(rad) | Leg RMSE600(rad) | Standing max joint/root600(rad/m) |
| --- | --- | --- | --- | --- |
| Nominal | 0.30669→0.39069 | 0.22076→0.19741 | 0.14732 | 0.26725 / 0.08482 |
| Standing push X | 0.26471→0.41295 | 0.22224→0.19925 | 0.15026 | 0.25029 / 0.10846 |
| Standing push Y | 0.25411→0.31428 | 0.22193→0.19898 | 0.14958 | 0.27478 / 0.07294 |

All3×1,841 controls complete, no reset/fallback, all landmark screens FAIL.
All leg lag40ms; no metric time shift. Source head p95 nominal/X/Y
0.43650/0.47964/0.36491m. Measured posture and legs improve across all3 cases,
but root tracking worsens across all3: this is a tradeoff, not overall success.
Source projection losses0.0643169/0.0637001/0.0605611rad², any projected means
50.50%/50.50%/46.75%. The complete original29 comparison also ran: original-source
world head p95 nominal/X/Y1.06280/0.96114/0.98834m, with no amplitude/lag/root
alignment. These are not the adapted-reference scores and do not prove parity.

One further unchanged-upper-objective block600→700 is running under
`campaign/segment_0600_0700_upper_posture_v2/`, same128×64×100 batching and fixed
physical/acceptance limits. This tests whether the first reward-transition
tradeoff settles. Do not continue this configuration through a second root
regression.58 focused tests pass again. The deferred whitespace-only Ruff I001
was fixed after all600 hash-bound export/evaluation jobs completed; scoped lint
passes. No training/retarget source is edited during its hash-bound job.

Full-excursion reference: weighted and hard-protected candidates both reject,
236/1,091 valid control frames. Overlapping failures: COM169, left foot
orientation346, right foot orientation536. Feet position/hard joint/trajectory
failure counts are zero. Strict solver's first linearized subproblem reports
PrimalInfeasible; this is not a physical impossibility certificate. Both failed
artifacts remain under `planned_dance_full_excursion_20260907_v1/` on C:.
Added explicit planned-source `--feasibility-restoration`, requiring a hash-bound
rejected diagnostic. Reuses the existing bounded intermediate solver; original
final task and temporal constraints unchanged.14 planned-option/restoration
tests and scoped lint pass. Candidate now running in `restored_protected/`.
No rejected or replacement reference has entered the policy training run.

## 2026-09-08 Sydney: buffered500 evaluated; upper-posture and full-range work active

Simulation only; hardware untouched. No robot, DDS, mode, motor, commit or push
actions. Goal unfinished: neither dance fidelity nor live teleop is qualified.
The400→500 block completed819,200 transitions, cumulative2,867,200 critic
samples. It traversed96 full reference timelines and616 sampled suffixes, with
128 standing and712 source resets. Full-state continuation and export checks
pass; encoder10 unchanged, decoder18/all36,864 root weights updated, maximum
root weight0.0390887. Checkpoint500 SHA
`f67183e4e3d51cc1c2cf104c271b137ad262c0b9ecaef75cbc1488c25e88029c`,
actor `98098e164ae3e66b34fce130ae7e5b19e10ebe6dbda7db22ff8cfc6a1110919b`,
lineage `d6230d6b91032cda231bdf02365499baeaf016d8b3801de9e6519c5778a7ff6d`.
Evidence: `C:/Users/camer/sonic23_sim_artifacts/received_source_horizon_20260907_v1/campaign/segment_0400_0500_projection_v1/`.

| Full CPU case | Root p95,400→500(m) | Leg RMSE at500(rad) | Head p95 at500(m) | Final standing max joint/root error(rad/m) |
| --- | --- | --- | --- | --- |
| Nominal | 0.47103→0.30669 | 0.15197 | 0.36136 | 0.36604 / 0.10861 |
| Standing push X | 0.42391→0.26471 | 0.15425 | 0.29722 | 0.36574 / 0.20075 |
| Standing push Y | 0.60620→0.25411 | 0.15352 | 0.29647 | 0.37386 / 0.06320 |

All3×1,841 controls finish without reset/fallback; all landmark screens still
FAIL. Leg lag40ms; all-joint RMSE nominal/X/Y0.22076/0.22224/0.22193rad.
Source-mean projection losses0.0645701/0.0876946/0.0778449rad²; any mean projected
54.63%/55.18%/55.73%. X projection loss worsens despite better root tracking.
Standing posture worsens for a second block, so unchanged training was paused.

Implemented `root_and_upper_posture_v2`, adding exactly one reward:
`-20 * mean_over_10_arm_joints((measured_joint_error/action_scale)^2)`.
Existing root and full23 posture objectives remain-10; no leg/waist penalty,
PD-action teacher, gain, cap, observation, encoder or acceptance-rule changes.
This explicitly changes the reward objective while preserving actor, critic,
optimizer/counters and the existing projection PPO loss.58 focused tests pass,
including real configuration equality for every pre-existing reward/action/
termination and rejecting nonfinite/wrong-shape states. The500→600 block is
running at128×64×100 under `campaign/segment_0500_0600_upper_posture_v2/`.
The actual constructed environment logs the new-20 upper-posture term.
One whitespace-only Ruff I001 in `g1_true23_root_feedback_objectives.py` remains;
defer that formatting edit until hash-bound600 jobs finish.

New read-only `measure_g1_true23_original_source_tracking.py` compares all1,091
measured source controls to all546 original planned29 frames through the declared
2x-duration interpolation map. It reconstructs original source arrays/FK,
checks complete lifecycle/trace/model hashes, and never shrinks the comparison
source or aligns it to the measured robot. World and pelvis-relative position
errors stay separate. It does NOT compare original29 closed-loop dynamics,
orientation accuracy, contact quality or original-speed parity.8 focused tests
pass; it ran for400 and500, preserving both reports under
`evaluate_updated/original29_source_tracking.json`.

This exposes a second gap: v5 retarget uses90% excursion, including horizontal
travel. Its *reference itself* has original-source foot/head world p95
0.83236/0.87630/0.83321m. Even a perfect policy cannot reproduce original travel
from that shrunken reference. At500, actual original-source head p95 nominal/X/Y
0.81832/1.04674/0.86380m (do not substitute the smaller adapted-reference scores).
The historical retarget XML and current evaluation XML differ only in CRLF/LF:
raw hashes38d6b065... and16e304c9..., identical normalized text and compiled model
SHA4965a4f65b5045aaa5e9509c0a90ea1629d0535ab234b56140baba561f8a5525.
The comparison binds both raw files and verifies compiled identity; it does not
rewrite old hashes or silently equate different geometry.

Added opt-in `--preserve-source-excursion` to planned-trace root refinement:
one2x-duration/100%-excursion candidate, unchanged solver/physical/fidelity
bounds, no implicit sweep. Existing90% behavior remains unchanged.16 tests pass
across planned options and source-comparison helpers. A new full546-frame fit
is running under
`C:/Users/camer/sonic23_sim_artifacts/planned_dance_full_excursion_20260907_v1/weighted/`.
It has not replaced v5 training data. If the weighted fit rejects, retain its
diagnostic and use the existing hard-protected refinement before any adoption.

Additional BONES-SEED reference: preselected train-split
`body_check_001__A439`,1,408 original120Hz samples, now fits1,173 controls at
50Hz with1.999147x duration and100% excursion. Fit AND independent all-original-
timestamp FK audit PASS. Accepted NPZ SHA
`f96a104776b1dafa3a06270c873eb8592ac3a2dc2f73a08e029dafe93a4a4412`.
Original-source hand/head p95 in this kinematic fit0.07571/0.07336/0.04321m.
Evidence: `C:/Users/camer/sonic23_sim_artifacts/bones_seed_extra_fit_20260907_v1/body_check_001__A439/`.
Not yet trained/evaluated dynamically; no held-out generalization or dataset
qualification claim. Source and derivative data remain local.

Measured400 video: `segment_0300_0400_projection_v1/evaluate_updated/nominal.measured-simulation.mp4`
under the C: experiment root,1,842 recorded states/50Hz,36.82s integrated.
SHA32c82e8db7235c6814794a1b12f6c5de62926d0a268a7c057fb26a7f2ccd3bdf;
traceSHAcd8baafe745f1f6cdc47ad22b0fbe6a7a0fc2c55a2230d45a558444b76bec8ed.
App opening returned queued, not confirmed played.18s frame was inspected.
Recorded state rendering is not policy/simulator qualification.

## 2026-09-08 Sydney: buffered400 improves world tracking; standing arms regress

Simulation only; no robot, DDS, mode, motor, commit or push actions. The first
projection-loss block completed100 updates/819,200 transitions, cumulative
2,048,000 critic samples. It traversed96 full reference timelines and636 sampled
suffixes;128 standing and736 source resets. Executed auxiliary-loss receipt
confirms100 PPO updates/800 minibatches. Full actor/critic/optimizer/counter
continuation and export checks pass: encoder10 unchanged, decoder18 changed,
all36,864 root weights changed; maximum root weight0.0339391. Checkpoint400 SHA
`7d18d0f2c0f4bba1777a15a919f4cc29dde556b04fc964e52bb739eebf3b041c`,
actor `71af06658c169d800d1ed36fcb4eea8660fb55616a28e60b239ddd141d2c8dfe`,
lineage `8057e3cbbddeecafa1f5f13a3721bd9cfd90d4dc5b054332c0b1f89228015f9a`.
Evidence: `C:/Users/camer/sonic23_sim_artifacts/received_source_horizon_20260907_v1/campaign/segment_0300_0400_projection_v1/`.

| Full CPU case | Root p95,300→400(m) | Leg RMSE at400(rad) | Head p95 at400(m) | Final standing max joint/root error(rad/m) |
| --- | --- | --- | --- | --- |
| Nominal | 0.80490→0.47103 | 0.15725 | 0.52975 | 0.27270 / 0.09216 |
| Standing push X | 0.49277→0.42391 | 0.15885 | 0.46990 | 0.27013 / 0.15518 |
| Standing push Y | 0.89431→0.60620 | 0.16345 | 0.65050 | 0.27841 / 0.04041 |

All3×1,841 controls finish without reset/fallback; all source landmark screens
still FAIL. Diagnostic leg lag remains40ms. World-position and leg metrics
improve in all3 cases, but standing joint error worsens. The standing regression
is principally elbows: nominal left/right RMS0.25889/0.23633rad, versus
0.19121/0.14215rad at300. Do not call this a deployment-ready standing handoff.

Recorded-state projection-loss values at400 nominal/X/Y are
`0.0714723/0.0748706/0.0797811 rad²` versus300
`0.0939434/0.0841445/0.0832933`. At least one mean remains projected in
57.20%/55.64%/55.91% of source controls. A separate exact-input replay applied
decoder400 to all3×1,091 *checkpoint300* source inputs, checking frozen encoder
outputs bit-for-bit. Its losses are0.0909531/0.0832723/0.0814274rad²: small
improvements, but projected-frame counts slightly increase. Hence the bigger
closed-loop reduction cannot be attributed solely to the auxiliary loss; this
block also performed ordinary PPO updates and changed the visited states.
Executed target changes on those identical inputs have about0.032rad RMSE.
No stronger exploration-effectiveness or dynamic qualification claim follows.

The next block may retain this objective because all3 world-tracking cases
improved, but it must also report the arm/standing regression, with no gain,
effort, reference, or acceptance-limit changes. Further changes must be judged
against both sets of metrics, not training reward alone. Added upstream RSL-RL
BSD attribution to the adapted PPO source after all400 jobs completed; no
executable logic changed by that attribution edit.

## 2026-09-08 Sydney: buffered300 evaluated; unreachable-mean PPO fix running

Still simulation only. No robot, DDS, hardware-mode, motor, commit or push actions.
The128-environment200→300 block completed819,200 transitions; cumulative critic
count1,228,800. It traversed92 full reference timelines and627 sampled suffixes,
with129 standing and733 source resets. These are not fidelity passes. Export and
actual full-state continuation checks pass; source encoder10 unchanged, decoder18
and all36,864 root-conditioner elements updated. Checkpoint300 SHA256
`59e3991fa60d7b85f1614cb329c3c1694609b951abec7db2abd4af5625fe83e5`;
lineage `784ecf056c47d8ac1092b9b2430a505588d048afaf3b7204a02e236626bac1fb`.
Evidence: `C:/Users/camer/sonic23_sim_artifacts/received_source_horizon_20260907_v1/campaign/segment_0200_0300/`.

| Full CPU case | Root p95, 200→300 (m) | Leg RMSE at300 (rad) | Head p95 at300 (m) | Final standing max joint error (rad) |
| --- | --- | --- | --- | --- |
| Nominal | 1.1132→0.8049 | 0.15986 | 0.82992 | 0.19658 |
| Standing push X | 0.9656→0.4928 | 0.16433 | 0.52885 | 0.16182 |
| Standing push Y | 0.7347→0.8943 | 0.16442 | 0.91123 | 0.19948 |

All3×1,841 controls finish without reset/fallback, all fidelity screens FAIL.
Diagnostic leg lag remains40ms. Root improves in two cases and worsens in Y;
leg accuracy does not materially improve. No simulator/deployment promotion.

The corrected BONES-SEED transition was also evaluated with checkpoint200 without
training on it: all3×1,491 controls finish, but all landmark screens FAIL. Root
p95 nominal/X/Y `0.166397 / 0.174423 / 0.145927 m`; head p95
`0.137070 / 0.174050 / 0.129763 m`. This is an unseen train-split motion probe,
not held-out split/generalization qualification. Kinematic acceptance remains
distinct from this failed dynamic-policy tracking result.

Read-only actuator check ruled out an accidentally retained quarter-effort cap:
nominal benchmark uses full configured88/139/35/25Nm effort limits, with no target
slew or quarter-effort projection. In checkpoint200 nominal, only ankle-pitch
torque saturates, for0.456%/1.152% of physics steps. No motor limits were changed.

A different learning bottleneck is measured in the recorded checkpoint200 nominal
source states. The decoder's ankle-pitch means exceed the *target-position*
envelope in18.15%/27.22% of controls, by up to30.12/22.00 standard deviations.
Using0.100006 as a conservative upper bound on that checkpoint's measured noise,
13.38%/20.16% of source states have <1% probability of an unprojected ankle-pitch
sample. These are evaluation-state diagnostics, not proof of training occupancy.
The Gaussian was exploring far outside the reachable envelope, where many
different samples execute the same projected target. This does not justify
removing the envelope or changing physical limits.

Implemented opt-in `source_target_projection_l2_v1`: explicit PPO auxiliary loss
`0.05 * mean_batch(sum_joints(squared_radian_mean_target_overreach))`. It pulls
only unreachable means toward their existing physical target projection. The
Gaussian, action likelihood/entropy, source encoder, actor/export architecture,
reference timing, physics, reward objectives and evaluation limits are unchanged.
This is a PPO objective transition, explicitly declared and bound in the new
lineage; it is NOT represented as an unchanged-loss resume. Actor/critic/optimizer
and learning rates are preserved. Initial weights are not manually clipped.

The local fixed-rate feedforward PPO implementation rejects unsupported recurrent,
multi-GPU, RND and symmetry configurations. With auxiliary weight0, four tests
match upstream PPO's actual losses and parameter updates bit-for-bit across
clipped/unclipped value loss and advantage normalization settings. A positive
weight moves unreachable means inward even with zero PPO advantages;64 random
probes match the existing source-action target projection. Focused integration
suite:70 tests pass, including ONNX parity and rejecting tampered loss contracts.
Scoped Ruff passes. This verifies implementation, not tracking improvement.

The evaluated300→400 projection-loss block is now starting under
`campaign/segment_0300_0400_projection_v1/` in the C: experiment root,128×64×100
=819,200 requested additional transitions. It will be judged by complete CPU
nominal/X/Y rollouts, source tracking/projection metrics, and standing return.

## 2026-09-08 Sydney: buffered200 evaluated; larger simulation batch running

Simulation only. No robot/DDS/mode/motor actions, deployment approval, commits,
pushes, or changes to protected hardware files. The goal is still unfinished.

Buffered100→200 completed: 32×64×100 = 204,800 additional transitions, 409,600
cumulative critic samples. This block traversed 24 full reference timelines and
148 sampled suffixes, with 34 standing and 175 source resets. Export and actual
continuation audits pass: actor/critic/optimizer/counters preserved, encoder10
unchanged, decoder18 changed, all 36,864 root-conditioner elements changed.
Checkpoint200 SHA256 `b6a242024c489553594b39e66c0fd77bf98a6aa802dd741a74830dd5b891ed3f`;
initial100 `877f522fbad48c2cd141eb22e27753eb02d5bbe2484aa626706525a2aa258c3d`;
lineage `e8a1a1c1113e0c204621f32c338c0c6ab279182b27ea925cd8672530fd9fd95f`.
Evidence: `C:/Users/camer/sonic23_sim_artifacts/received_source_horizon_20260907_v1/campaign/segment_0100_0200/`.

| Full CPU case | Root p95, 100→200 (m) | Leg RMSE at 200 (rad) | Head p95 at 200 (m) | Final standing max joint error (rad) |
| --- | --- | --- | --- | --- |
| Nominal | 1.1543→1.1132 | 0.15809 | 1.09585 | 0.13435 |
| Standing push X | 0.8789→0.9656 | 0.16445 | 0.95475 | 0.19221 |
| Standing push Y | 1.3663→0.7347 | 0.16299 | 0.69663 | 0.15307 |

All three integrate 1,841 controls without resets/fallback, but all tracking
screens FAIL. Leg lag remains 40 ms; leg RMSE is slightly worse than100 in all
cases. Root/head improve in two cases, worsen in X. These results do not qualify
teleop, standing return, or arbitrary dance generalization.

The next evaluated200→300 block was launched in `campaign/segment_0200_0300/`.
An explicit `--campaign-batching parallel128_v1` increases only local simulation
resources: 128 environments×64 steps×100 updates = 819,200 requested transitions.
The old default remains32, smoke/regression limits remain unchanged, and the
independent CPU evaluation interval remains at most100 updates. Timing,
objective, reference, native23 model, gains, torque/joint limits, learning rates,
and acceptance thresholds are unchanged. The new batching dimensions are
hash-bound in training metadata; this is not a matched-batch ablation. Focused
training/continuation/buffer/retarget tests: 61 passed; scoped Ruff passed.

Independent data work: added a bounded offline original-time foot-orientation
repair for accepted BONES-SEED control-grid fits. It retains all original source
frames and only changes selected affected-leg knots within0.01 rad; both full
control-grid and original-time FK audits must pass after float32 serialization.
The first ankle-only fit did NOT fix the two failures and was retained as a
reject under `C:/Users/camer/sonic23_sim_artifacts/bones_seed_original_time_repair_20260907_v1/`.
A second fit allows the affected leg's hip/knee/ankle chain, preserving root and
all other joints. Under `..._v2/`, it passes both full FK grids with a maximum
joint correction of `0.00018365681171417236 rad`, but SLSQP reaches its80-iteration
budget. The first implementation therefore retained it as rejected despite
zero independent physical/task violations. Its optimizer constraint minimum
was `-5.2722432571562155e-9`, against an extra fit-only orientation margin of
2e-6 rad; this margin is not the source-fidelity tolerance.

Version3 reruns the fit and separates verified feasibility from optimizer
convergence/optimality. Publication requires all original FK gates, all control
trajectory gates, finite serialized poses, unchanged unselected/root components,
and the unchanged0.01-rad correction bound. It explicitly does NOT claim minimum
norm optimality or hide the optimizer's iteration-limit status. All checks pass.
The existing independent source-timestamp auditor then reread the published
bytes and recomputed890/890 original frames successfully, with no foot-orientation
excess and no failures. 741 control frames retained, unchanged duration/excursion.
Accepted motion SHA256 `076cba537f9212c4e013f64306d743fdd40315ecc52c9c1bda1bd42ae79a270b`.
Evidence: `C:/Users/camer/sonic23_sim_artifacts/bones_seed_original_time_repair_20260907_v3/idle_right_to_idle_R_002__A026/`.
The motion is kinematically accepted, NOT dynamically or hardware qualified.
It remains a train-split source never yet used to train this policy, not a held-out
split claim. All three complete lifecycle CPU cases with buffered checkpoint200
are now running in its `policy200_unseen_evaluation/` subdirectory.

The expanded independent original-time audit/repair tests pass35 cases, including
real-FK correction and rejection of edits outside the selected joints, nonfinite
poses, excessive correction, or failure of either full FK gate. One temporary
test-file syntax error was fixed before this passing run. Scoped Ruff passes.
Earlier28/61 counts overlap this set; they are not additive.

128-environment training construction was also verified against the received-only
buffer: lower/VR bit exact, encoder max error `2.922024577856064e-8`. Observed
throughput is about1,300 transitions/s versus roughly300–360 for32 environments,
with a sampled GPU allocation of1,226 MiB/8,192 MiB during collection.

## 2026-09-07: update 400 evaluated; explicit received-source timing experiment

Simulation only; no hardware commands or deployment approval. The corrected
nominal-physics mixed-reset campaign completed 300→400: 32 environments × 64
steps × 100 updates = 204,800 new transitions, 512,000 cumulative. Actual
continuation audit passes: actor/critic/optimizer/counters exact at transfer,
10 encoder tensors unchanged, 18 decoder tensors changed, all 36,864 root
conditioner elements changed. It traversed 23 full reference timelines and 152
sampled suffixes; those are training completions, not tracking passes.

Checkpoint SHA256 `1e935caa8166b41497af8bb9824c5bb1385a5fd460dd128b70f6fc4f0293e115`;
lineage `bf02b13e97de5a34694535387675c09dc85fbb072759c6b3ba3e30f05fdda670`.
Evidence: `nominal_scene_mixed_campaign_20260907_v1/segment_0300_0400/`.

| Full 1,841-control CPU case | Root p95 (m), update 300→400 | Head p95 (m), 400 | Dance fidelity |
| --- | --- | --- | --- |
| Nominal | 0.3762→0.4315 | 0.4750 | FAIL |
| Standing push X | 1.2184→0.5610 | 0.6105 | FAIL |
| Standing push Y | 0.9608→0.6597 | 0.7171 | FAIL |

All cases complete without resets/fallback. Nominal worsened; push cases
improved. This is not robust deployment qualification. No blind 400→500
continuation was launched; the next experiment targets a measured timing issue.

Measured simulation video (not commanded reference animation):
`nominal_scene_mixed_campaign_20260907_v1/segment_0200_0300/evaluate_updated/nominal.measured-simulation.mp4`.
1,842 measured states, 50 Hz, 36.84 s video including initial state; video SHA256
`c6097db9666101b8650879d0a5039b52933826a959b461932c5e70c0fbbedce7`.
Frame at 18 s viewed; video preview was queued in Codex. Upright appearance does
not override failed tracking metrics.

Read-only lag diagnosis: old matched causal initial leg RMSE 0.2259 rad, best
diagnostic lag -10 frames (200 ms); same old saved-preview comparison 0.1635 rad,
best lag -2 frames. Updated300 causal RMSE 0.2194 rad, again -10 frames. These
are diagnostic cross-correlations only: no scored trajectory was time-shifted.
The preview comparison predates corrected CPU cached constants; it identifies
a timing hypothesis, not a matched current-physics deployment improvement.

New `gear_sonic/teleop/buffered_source_horizon.py` consumes 11 received 50-Hz
samples. It supplies source-order q0..q9 positions and q0→q1..q9→q10 forward
differences, q0 virtual-source VR, and CURRENT measured robot orientation/state.
Encoder anchor latency is explicitly 200 ms; next root setpoint is 180 ms old.
It rejects missing/duplicate/stale/nonfinite samples and latches until explicit
reference-buffer reset. No robot transport, future prediction, EOF padding,
motor commands or controller fallback. Its 14 standalone tests passed.

Opt-in `--reference-timing received_source_horizon_200ms_v1` now has a distinct
training/actor/root-feature/export/runtime contract. Old causal artifacts cannot
be relabelled. Fresh initialization is required for the first bounded test.
The offline scripted standing source transmits ten explicitly generated extra
samples after the scored lifecycle, solely to keep the reference buffer fed;
original dance samples, scoring times, scored duration and physical limits stay
unchanged. These generated inputs are not represented as captured teleop data.
Focused integration suite: 99 tests passed. A second, partly overlapping set
of 19 tests passed, including executing original SONIC lower-body position and
velocity properties and rejecting old contracts even with rehashed lineage.
Three new read-only phase-measurement tests pass. Scoped Ruff checks pass.

Actual buffered GPU smoke: 4 environments, 2 updates, 64 transitions. Constructed
encoder matches the independent received-sample implementation: lower240/VR21
bit exact, orientation maximum error `1.1175870895385742e-8`. Full32-environment
construction check also passes (`1.6065314412117004e-8` maximum). All 23 physical
action slots remain present. Profile SHA256
`8006caf5e51300903171e1d2965fd8afca21d7bbebfdf441cdc1538b944870f4`, compatibility
`308a6d89fedc33724f14fcf41ca0a6775986a2edc4581e47339b3866240c82e2`.

New experiment artifacts are on `C:/Users/camer/sonic23_sim_artifacts/received_source_horizon_20260907_v1/`
(WSL `/mnt/c/Users/camer/sonic23_sim_artifacts/received_source_horizon_20260907_v1/`)
because Z: has ~4.2 GB free. Earlier evidence was not deleted or moved.
Smoke checkpoint0 SHA256
`e72a28668a933b0b836bc81bf6306946b3df52f430b278b08f8b810c29022304`;
smoke and regression checkpoint0 actor tensors and contracts are independently
verified identical, actor SHA256
`d22dabc4f0467e616dd3cc32dc22ee5327d4446697ec947d24a87fd1afdef122`.
Regression lineage `e7c6d2b941934dbea01c37a2f7dacaa9d4e8368e7d0f6b01a8e751a622ed4111`.
The 32×64×100 fresh regression completed: 204,800 transitions, 21 full reference
timelines and 159 sampled suffixes. Actual update audit passes all checks:
source-exact initial actor, encoder10 unchanged, decoder18 changed, 36,864 root
conditioner elements changed (maximum absolute weight 0.0130678033), bounded
exploration, finite optimizer/critic, nonzero action response to all nine root
features. Checkpoint100 SHA256
`64608a23907a74a2a3b1b0d7d7817f0c314847bb34b92d23dff52f9aca83f2f8`;
actor `b8fdb7f7c5d5de6781a2f9abf2c5b95fa004d024ae849c50329e2a0456302825`.

Buffered zero-update ONNX export passes exact token parity; decoder maximum
error `1.1920928955078125e-6`. All three full 1,841-control CPU cases completed,
but all fidelity screens fail. Root p95 nominal/X/Y:
`1.511626 / 1.579117 / 1.525393 m`. Full dance leg RMSE:
`0.164763 / 0.167804 / 0.166048 rad`; all three diagnostic lag estimates are
40 ms. Thus the timing hypothesis is supported, but drift is not fixed by
buffering alone. `smoke/evaluate_initial/tracking_phase.json` preserves full
unshifted errors separately from interior-window diagnostic cross-correlation.
The earlier fixed-window nominal diagnostic gave 0.165417 rad; different window,
not a discrepancy. Simulated emitted-minus-anchor timestamps verify 200 ms.

Buffered checkpoint100 export passes. All three complete CPU cases still fail
fidelity, but every case improves root, head and leg metrics versus its exact
zero-step actor. Full unshifted source metrics, unchanged acceptance rules:

| Buffered case | Root p95, 0→100 (m) | Leg RMSE, 0→100 (rad) | Head p95 at 100 (m) | Final standing max joint error (rad) |
| --- | --- | --- | --- | --- |
| Nominal | 1.5116→1.1543 | 0.1648→0.1580 | 1.1705 | 0.1474 |
| Standing push X | 1.5791→0.8789 | 0.1678→0.1635 | 0.8934 | 0.1562 |
| Standing push Y | 1.5254→1.3663 | 0.1660→0.1593 | 1.3886 | 0.1559 |

All diagnostic leg lags remain 40 ms. By comparison, causal400 full-source leg
RMSE is 0.2200/0.2168/0.2174 rad with 200 ms lag, but its root errors are smaller.
These are different training budgets/semantics, not a matched-budget superiority
claim. The buffered root error remains too large for deployment.

Because this first bounded block improved all three cases, an evaluated
100→200 continuation was launched under `campaign/segment_0100_0200/` in the C:
experiment root. No objective, timing, model, motor limit or scored reference
change. New campaign support rejects mismatched timing/force contracts and
re-feeds every saved source packet against actual measured qpos/qvel: 5,523
controls verified across the three cases, including exact lower/VR channels,
root feedback and source timestamps. Generated terminal source inputs are
independently reconstructed from the complete lifecycle and original29 FK,
then hash-bound into the continuation receipt. This is input verification,
not motion qualification. The focused continuation/test set passes 49 tests.

## 2026-09-07: original-source action units audited; v2 training interface implemented

Still simulation only. Physical dance/live teleop remain unqualified. No robot,
DDS, mode, motor, deployment-file, commit or push action was taken. The blocked
goal UI was not falsely completed and does not prevent work in this turn.

Two source-convention discrepancies were independently confirmed:

- V1 `released_bounded_linear` used **native23** action scales. Original SONIC
  low-latency config selects `g1_model_12_dex`; the associated training source
  computes hip-pitch scale `0.3506614663788243`, while native23 uses `0.55`.
  V1 therefore requests 1.56846 times the original hip-pitch displacement for
  equal raw outputs. This is not proof that scaling alone causes dance failure.
- Historical Python `RELEASED_RETAINED_KD` has ankle damping `0.90722`; actual
  original C++ is `1.81445`. Historical results keep their old identity. A new
  `original_cpp_gains_diagnostic` compiles the original header and uses captured
  float32 gains with native23 effort caps; it is **not nominal qualification**.

New `g1_true23_source_action_codec.py` separates original output units from
native inverse-tanh scales. Source target angles are projected only into the
unchanged reachable envelope. Previous-action observations convert the effective
target back to original units; measured pose/velocity and absent slots stay
untouched. Original raw-history equality applies only to unprojected targets.
Native global scale tables, motor gains, armature, effort limits, physical
geometry and acceptance thresholds were not changed.

Independent original C++ one-hot oracle passes 47 probes: retained29-to23 mapping
and default angles exact; maximum target difference `4.90e-8 rad`, normalized
history difference `2.24e-8`. Pinned source checkpoint/config, original training
robot definitions and C++ files are recorded in
`source_action_codec_audit_20260907_v2/report.json`. Its earlier v1 report predates
a formatting/input-stability-check change; v2 is the completed stable audit.

Zero-training full-lifecycle tests all retain 1,841 controls/1,091 dance frames,
no pose rewrite after reset, no fallback and no physical-bound relaxation:

| Explicit runtime | Full-lifecycle root p95 (m) | Outcome |
| --- | --- | --- |
| Source-scaled v2, causal, native motors, nominal/X/Y | 0.661 / 1.121 / 0.604 | All dance screens fail |
| Source-scaled v2, future preview, native motors | 1.958 | Dance screen fails |
| Source-scaled v2, future preview, historical Python gains | 1.388 | Dance screen fails |
| Source-scaled v2, causal, exact C++ gain counterfactual | 0.401 | Dance screen fails |
| Source-scaled v2, future preview, exact C++ gain counterfactual | 1.488 | Dance screen fails |

Evidence directories: `source_action_semantics_20260907_v1/`,
`source_action_preview_20260907_v1/`, `source_action_exact_cpp_gains_20260907_v1/`.
Future preview uses saved future frames and cannot qualify live teleoperation.
The first two reports retain a generic native-history label from the referee;
their explicit adapter contract and actual decoder/history traces document the
source-unit conversion. New reports label consumed versus stored history
separately. No previous report was rewritten or promoted.

Full nominal-effort inverse-force hypothesis is in
`planned_v5_nominal_support_20260907_v1/report.json`: 1,852 poses checked, 1,251
without candidate floor contact within 2 mm, only 103 conditional force solutions.
Source phase has 575/1,091 without contact and 66/1,091 conditional force solutions.
This is an exact-pose necessary-condition diagnostic with approximate contact
cones/pose derivatives, not a proof that nearby trackable motions are impossible.
Independent distance checks show standing gap 3.00 mm, dance gap range
`-20.14..31.82 mm`, dance p95 11.48 mm; no dance foot gap exceeds 50 mm. These small
gaps cannot alone explain the observed large horizontal drift. Source reference
has kinematic acceptance, **not contact/dynamics acceptance**.

Training/export/CPU continuation now support explicit
`--release-action-convention released29_scale_bounded_linear_v2` together with
`--release-source-geometry`. V2 has a distinct hashed compatibility contract and
actor identity; v1 contract digest remains unchanged. Native action buffers are
not repurposed: actor previous-action observations are converted before history
collection. Old v1 checkpoints cannot be relabeled/continued as v2. Independent
CPU evaluation must execute the matching codec, and gain/model counterfactuals
cannot replace nominal campaign evidence. Focused existing suites: 65 tests pass;
training/export/codec suite 60 pass; updated compatibility suite 14 pass.
Constructed two-update v2 smoke completed under
`source_action_training_smoke_20260907_v1/`: four environments, 64 transitions;
GPU targets agree with CPU within `4.77e-7 rad`, source-normalized previous
actions within `9.54e-7`; virtual q9 references exact.

Fresh v2 100-update run completed under `source_action_regression_20260907_v1/`:
16 environments, 51,200 transitions, 16 completed reference timelines and 32
reset samples, same synchronous schedule/budget as the earlier fresh v1 run.
Checkpoint 100 SHA256:
`447a209b1da66c3ba1c0562ee553d5db39425e449a6f32755f9d1e763167ff1a`.
Independent update audit passes: ten encoder tensors exact, all 18 decoder
tensors changed, all 36,864 root-conditioner elements changed, critic/optimizer
finite. The first audit attempt hit host memory pressure while two CPU
evaluations ran; sequential retry completed. Export token parity exact; decoder
maximum probe error `1.19e-6`.

Matched ONNX evaluation completed all 1,841 controls in all six cases:

| Root-position p95 (m) | Initial | 100 updates |
| --- | --- | --- |
| Nominal | 1.471 | 1.726 |
| Standing X push | 0.684 | 0.874 |
| Standing Y push | 1.254 | 2.188 |

All dance-fidelity screens fail. Nominal updated source-phase head p95 is
1.377 m, hand p95 1.436/1.367 m, ankle p95 1.352/1.452 m; final standing maximum
joint error 0.279 rad. **Rejected; no v2 100-to-200 continuation.** Exact target
unit correction is necessary interface work, not a successful dance policy.
The initial ONNX and earlier initial Torch closed-loop trajectories also differ
substantially despite small inference-probe differences; neither is qualified.

Next implemented change is opt-in `pinned_cpu_referee_scene_v1` training:
use the pinned native23 MJCF/meshes instead of the old capsule asset; retain
the CPU referee's contacts, root passive resistance and solver. Motor action
clipping and joint-force limits remain unchanged; CPU evaluation/model files
and acceptance thresholds are not edited. New constructed-model and same-state
physics checks are being run before any further training. Existing checkpoints
cannot be continued while silently changing this training physics profile.

The strict same-state check found an additional initialization bug in
`prepare_mujoco_model`: native armature was overwritten **after compilation**
without refreshing MuJoCo's derived `dof_M0`, inverse-weight and actuator
acceleration constants. A construction-time `mj_setConst` now refreshes them;
the physics contract explicitly records this operation. No XML, configured
motor value or acceptance threshold changed. This does change old CPU dynamics,
so earlier failed runs retain their historical compiled-model/source hashes;
they are not silently reclassified as current-physics comparisons.

New nominal scene passes ten perturbed-state, ten-substep CPU comparisons at
strict `1e-9 qpos` / `1e-8 qvel` tolerances. Actual GPU check in
`nominal_scene_parity_20260907_v1/` passes five saved states spanning the dance:
CPU constructed-training versus corrected-referee next velocity is **exact**;
Warp maximum difference is `7.15e-6`, with identical contact counts. This is
five 2-ms-step numerical evidence, not completed-dance qualification.
Construction also checks all 33 collision geometries and derived constants.

Initial focused test run: 47 passed; one older approval test fails because its existing
`APPROVED_CONFIG_SHA256=1ffce4...` does not match the current config `bc4dab...`.
The constant is identical in HEAD. It was not updated to manufacture approval;
hardware/promotion remain false. Expanded focused tests finish at 48 passed,
one explicitly deselected older approval-pin test; a subsequent new campaign
physics-transition rejection test also passes independently. Counts overlap
other suites and must not be added. Scoped Ruff and diff whitespace checks pass.
Initial v2 ONNX nominal re-evaluation after constant refresh still fails dance
tracking (full root p95 1.260 m), recorded separately in
`source_action_regression_20260907_v1/evaluate_initial_refreshed/`.
The corresponding old 100-update v2 actor also fails under corrected CPU
initialization: full nominal root p95 2.815 m, all 1,841 controls retained in
`evaluate_updated_refreshed/`. Historical old-physics reports are unchanged.

Constructed nominal-scene smoke completes two updates/four environments/64
transitions. Fresh nominal-scene 100-update training then completes 51,200
transitions, 16 full reference timelines and 33 reset samples under
`nominal_scene_regression_20260907_v1/`. Checkpoint 100 SHA256:
`187ef30856fc7be35f8e96789f1f9b2884993692e841be9b6c967f92458e0ad0`.
Actual update verification passes (encoder exact, decoder and conditioner
changed, finite critic/optimizer). Both initial and updated ONNX exports pass.

Matched corrected-physics CPU evaluation retains all 1,841 controls in six runs:

| Root-position p95 (m) | Initial | Nominal-scene 100 |
| --- | --- | --- |
| Nominal | 1.260 | 1.643 |
| Standing X push | 1.136 | 1.062 |
| Standing Y push | 1.237 | 1.848 |

All dance-fidelity screens still fail. Final standing maximum joint error
improves from 0.250/0.249/0.268 rad to 0.195/0.202/0.202 rad, but root tracking
worsens in two cases. Corrected physics is verified interface progress, not a
qualified dance or justification for repeated identical continuation.

The next bounded experiment changes only reset sampling: 100-to-200 updates
with `mixed_reference_reset_v1` on the corrected nominal scene, under
`nominal_scene_mixed_campaign_20260907_v1/segment_0100_0200/`. Training completes
51,200 additional transitions, four full standing-start reference timelines,
37 sampled source suffixes, eight standing resets and 51 source resets.
Suffixes are explicitly not full-dance qualification. Checkpoint 200 SHA256:
`fcd2a611e645eeb570c6609e6767997f95c9453351b77f1fd716144a7ac2e512`.
Independent continuation audit passes: parent actor/critic/optimizer/counters
are exact at transfer; encoder remains exact and decoder/conditioner update.
Export token probe parity is exact, decoder maximum error `1.19e-6`.
Full nominal/X/Y standing-start CPU evaluation completes all 1,841 controls.
Root p95 is 1.613/0.667/0.855 m versus parent 1.643/1.062/1.848 m. All dance
screens fail: nominal source-phase head p95 worsens 1.211 to 1.470 m, despite
the two perturbed cases improving to 0.661/0.704 m. Final standing maximum
joint errors are 0.221/0.240/0.214 rad. No promotion or monotonic-progress claim.
Measured nominal endpoint is [8.2363,-1.6903,0.7638] versus planned
[8.0836,-0.0848,0.7600]; the large error is mainly lateral, not height loss.
Pelvis-relative dance ankle p95 still 0.245/0.269 m and hand 0.195/0.191 m:
subtracting root translation does not make dance fidelity pass.

Completed next evaluated block, `segment_0200_0300`, with 32 environments and
64 rollout steps: 204,800 additional transitions (four times the prior block).
This deliberately changes rollout sampling/batch size, not physics, references,
motor limits or acceptance. It completes 23 full reference timelines and 149
sampled suffixes (suffixes not full-lifecycle evidence). Checkpoint 300 SHA256
`70c732c880250d29745ce01f5c5bd252325efe3ebc62262f0585af7e5cfa997a`.
Independent continuation audit passes, including exact initial actor/critic/
optimizer/counters and unchanged encoder; export passes. Full CPU evaluation
completes all 1,841 controls per case:

| Corrected scene, mixed starts | Root p95 nominal/X/Y (m) | Source head p95 nominal/X/Y (m) |
| --- | --- | --- |
| 200 (16 envs, 32-step rollout) | 1.613 / 0.667 / 0.855 | 1.470 / 0.661 / 0.704 |
| 300 (32 envs, 64-step rollout) | 0.376 / 1.218 / 0.961 | 0.372 / 1.024 / 0.807 |

Nominal improves substantially; both perturbed cases worsen. Worst-case and
mean errors across all three improve, but every dance-fidelity gate still fails.
Final standing joint maximums are 0.204/0.198/0.193 rad. These are provisional
diagnostics, not robustness or deployment qualification. Continue one further
bounded 300-to-400 block with the same 32x64 sampling and unchanged objectives,
then independently evaluate all three cases again. No hardware operation.
Reference joint positions all fit the unchanged finite safe-target envelope.
Prior nominal 200 policy's target projection affects 46.1% of controls, mainly
ankles (left/right ankle pitch 248/477 controls, maximum projection 1.358/1.167
rad). This is policy-request saturation, not proof of a static joint-range
infeasibility or permission to enlarge bounds.

### Additional untrimmed29 source-policy control (not a native23 result)

The old original29 baseline uses the normal release's G1 encoder, whereas
native23 starts from the pinned low-latency teleop encoder. Added
`audit_g1_sonic_low_latency29_baseline.py` to run the latter with all 29 joints,
the same saved original 546-frame dance, compiled model and captured C++ motor
parameters. It consumes nine saved future frames and cannot qualify live teleop.

The first diagnostic, `low_latency29_source_control_20260907_v1/`, is **invalid
as a source-policy comparison**: it reused older Step1B helper
`LOWER_BODY_IL29_INDICES`, which sorts/interleaves the legs. Actual original
`TrackingCommand` selects hardware `range(12)` mapped to IL29: left six then
right six. Its two initial-state labels also described identical states; they
are not independent trials. The native23 causal training path already uses the
correct left-six/right-six order. The older Step1B helper was not changed while
hash-bound training was active; do not use it as an exact source-reference oracle.

Corrected v2 uses hardware leg order; a regression test executes the original
TrackingCommand property's source AST and verifies its output. Seven focused
tests and scoped Ruff pass. Actual source29 and saved deployment-model FK
position/quaternion input features are identical on all 546 reference frames.
`low_latency29_source_control_20260907_v2/` completes 546/546 controls, no reset:
pre-control root p95 1.535 m; joint RMSE 0.2033 rad (post-state versus held
reference, 20-ms offset retained); minimum height 0.6810 m; travel 7.382 m versus
planned 8.975 m. Original normal-release comparison is 0.1858 rad and 8.384 m.
V2 trace SHA256:
`0ee036ba3a59b9477bb54bb4e3a03acc9abb02772d79cd105c2545a83052131e`.
No false source-checkpoint failure or native23 readiness claim remains.

## 2026-09-07: corrected semantics now trained and independently evaluated; not deployment-ready

Work continued despite the goal UI's blocked status. No goal was falsely marked
complete. No robot/network/motor commands, deployment edits, commits or pushes.
Existing dirty gantry work remains separate.

Implemented opt-in `--release-source-geometry` in root-feedback training:

- Frozen encoder references use original29 FK with absent axes fixed to zero,
  selected at the received q9 anchor. Only the reference's 21 VR values change;
  measured state, native23 physics, reward geometry and evaluation truth do not.
- The batched Torch action conversion matches the CPU bounded-linear inverse
  transform, with the same raw bound, inner joint envelope and motor effort.
  Previous-action observations contain the effective normalized target.
- Corrected actors carry distinct `native23_causal_release_compatibility_v1`
  metadata through checkpoints, ONNX and CPU evaluation. Legacy runtime loading
  rejects this profile unless its matching geometry/action path is explicitly
  selected. Legacy actor initialization cannot be relabeled as new training.
- Evaluated continuation requires matching semantics in parent, actual CPU
  runtime and next training segment; optimizer/critic/counters are preserved.
  Each segment remains bounded to 100 updates before a new full CPU evaluation.

Actual constructed GPU-environment parity passed 128 action probes:
maximum target difference `3.58e-7 rad`, normalized-history difference `9.54e-7`,
q9 virtual-reference values bit-exact with CPU. The check does not step physics
or change measured joint state. Fresh 4-environment/two-update smoke also passed.

Fresh 16-environment, 100-update PPO run completed 51,200 transitions from the
exact row-trimmed released SONIC initialization, with the measured-posture
objective. Receipt records 16 completed reference timelines and 32 reset samples
(initial reset plus one completed-cycle reset per environment), with all 16 at
control index 1359 after the second cycle's partial rollout. This establishes
training continuity, **not accurate dancing**. All 18 decoder tensors and all
36,864 root-conditioner elements changed; all ten encoder tensors stayed exact.
Critic/optimizer were finite; independent update audit passed. Export token
parity was exact; decoder maximum probe difference was `5.96e-7`.

Matched ONNX CPU comparisons use the complete V5 planned-endpoint lifecycle,
1,841 controls including all 1,091 source frames. All six initial/trained cases
completed with no diagnostic stop, but **all six dance fidelity screens fail**.
These are matched ONNX comparisons, not a substitution for the earlier Torch
zero-training ablation (small numerical backend differences change trajectories).

| Case | Root p95, initial -> update100 (m) | Dance head p95 (m) | Final standing max joint error (rad) |
| --- | --- | --- | --- |
| Nominal | 0.432 -> 0.480 | 0.487 -> 0.393 | 0.234 -> 0.242 |
| X push | 0.446 -> 0.431 | 0.536 -> 0.263 | 0.236 -> 0.253 |
| Y push | 0.304 -> 0.608 | 0.356 -> 0.394 | 0.255 -> 0.243 |

Nominal hands improve `0.516/0.512 -> 0.396/0.419 m`; ankles remain
`0.498/0.466 m`, far above unchanged `0.050 m` screens. Immediate two-second
push recovery root error improves X `0.0284 -> 0.0097 m`, Y
`0.0287 -> 0.0059 m`, but later Y-push dance drift regresses. No overall robust
tracking improvement or promotion is claimed.

Evidence: `artifacts/g1_true23_generalist/release_compatible_regression_20260907_v1/`
contains `train/`, `export_initial/`, `export_updated/`, `evaluate_initial/`,
`evaluate_updated/`, and `update_verification.json`. Update100 checkpoint SHA256:
`949b2717bb7da6b945a9287803e7cf0e711341a4d36de0467a161fe91095ab91`.

98 existing root/ONNX/CPU checks passed; latest 48 focused compatibility,
continuation, launcher and update checks passed. One intervening run reported a
mesh read error (not a logic assertion); the nonempty original mesh was verified
unchanged and the isolated test plus full focused rerun passed. No mesh was edited.

Continuation 100 -> 200 completed under
`release_compatible_campaign_20260907_v1/segment_0100_0200/`, using the full
update100 CPU report as its required parent evaluation. Full-state continuation
audit passed, but the fixed update200 policy **regressed and is rejected**:
nominal/X/Y full-lifecycle root p95 errors are `1.156 / 0.414 / 1.064 m`.
All cases complete 1,841 controls but fail unchanged landmark tracking screens.
Higher training episode reward (`-59.32 -> 31.64`) is not deployment evidence;
those episodes used changing weights and phase-synchronized environments.
Update200 checkpoint SHA256:
`9848f709f40ea97595cd34be475588c0c2da26cb78e7c00099afc013eae7762c`.

An actual same-state dynamics audit of the corrected training environment is in
`release_physics_parity_20260907_v1/`. CPU MuJoCo and Warp using the same training
model agree to at most `2.27e-5` in next-step generalized velocity over the five
sampled states. Training versus original CPU replay contact models differ by up
to `0.1064` in next-step generalized velocity: foot capsules versus spheres,
extra CPU free-root passive terms, and different solver settings. Masses and
driven-joint parameters agree. This revisits, rather than supersedes, the earlier
training/replay model-gap evidence below. Root/solver-only changes do not remove
the contact discrepancy.

Full update200 CPU tests on the exact compiled training model are preserved in
`segment_0100_0200/evaluate_training_model_v2/`. Nominal/X/Y root p95 errors are
`0.778 / 0.961 / 1.058 m`; all full timelines complete but all tracking screens
still fail. Therefore contact mismatch is **not the sole cause**. This explicit
counterfactual preserves native23 kinematics, masses, driven-joint limits and
integration cadence; it is not original-model qualification. Campaign loading
rejects counterfactual results as replacement parent evaluation. The initial
`evaluate_training_model/` failed before physics on an overly strict body-ID
check; v2 explicitly maps the extra static terrain body and namespaced robot.

Next isolated experiment implements `--start-schedule staggered_standing_start_v1`:
delay the first reference advance by `floor(env_id * lifecycle_controls / num_envs)`
standing controls. Each environment still starts standing; physical state/history
advance throughout the hold; no source frames are skipped or pose-written inside
episodes. The delay is applied only on the first reset, so later cycles do not
repeatedly add idle standing. Episode timeout covers delay plus the full lifecycle.
The schedule transition is explicit in lineage and keeps actor, critic, optimizer,
source arrays and primary CPU tests unchanged. Branch from the better evaluated
update100, not the regressed update200. Initial focused suite: 87 tests passed;
new 4-environment/two-update constructed training smoke passed under
`release_staggered_smoke_20260907_v1/`.

The staggered update100 -> 200 experiment also completed 51,200 new transitions
under `release_staggered_campaign_20260907_v1/segment_0100_0200/`. All 16 reference
phases are now distinct, all assigned holds executed exactly, 12 full timelines
completed, and only 28 resets occurred (16 initial + 12 completed timelines).
Actual weights/optimizer/counters were preserved at continuation, all ten encoder
tensors remain unchanged, all 18 decoder tensors changed, and state is finite.
Export parity passes (encoder exact; decoder maximum `2.15e-6`). Checkpoint SHA:
`68474a2743f10a47dadd9293749e1b241ad627f6bf250d5a5969c45647be57c5`.

Full nominal/X/Y CPU tests all complete 1,841 controls. Full-lifecycle root p95:
`0.392 / 0.543 / 0.489 m`. Final standing joint maxima improve to
`0.141 / 0.154 / 0.129 rad`, but source-motion head p95 worsens against update100
to `0.479 / 0.597 / 0.484 m`. All source landmark screens still fail. This is an
improved standing-return learner, **not an accurate dance policy**. Extra initial
standing time makes raw episode reward incomparable with synchronized runs.
113 focused regression tests passed after this experiment. No promotion.

Next isolated schedule: `mixed_reference_reset_v1`, starting again from evaluated
update100 for an equal-budget comparison. One quarter of environments retain full
standing starts; the rest sample initial q10 poses/velocities from original source
frames at environment reset, then track every remaining frame through standing
return. It targets the persistent source-tracking gap rather than adding more idle
standing. Sampled reference states are not asserted dynamically feasible; failures
remain training failures. Sampled suffix completions have a distinct counter and
never count as full lifecycles. Primary CPU evaluation remains unchanged, including
full standing start, every source frame, and standing return without pose rewrites.
New sampler tests initially passed 49 focused checks; constructed four-environment
two-update smoke passed in `release_mixed_reset_smoke_20260907_v1/`.

Mixed-reset update100 -> 200 completed 51,200 transitions in
`release_mixed_reset_campaign_20260907_v1/segment_0100_0200/`. Receipt separates
4 completed standing-start lifecycles from 37 completed sampled suffixes; 57
resets equal 16 initial + 4 + 37, with source samples spanning all ten deciles.
Full-state continuation/update audit passes; ten encoder tensors remain exact,
18 decoder tensors change and optimizer/critic remain finite. Export token
parity exact, decoder maximum difference `9.54e-7`. Checkpoint SHA256:
`4a8e7c631e2539843ab19b3f9cc1726e0070f1141ae84ab8ce2ec971f17c2053`.
All nominal/X/Y CPU cases complete 1,841 controls but fail dance fidelity;
root-response p95 is `0.835 / 0.679 / 0.564 m`. Nominal dance head p95 is
`0.599 m`, hands `0.668/0.660 m`. This checkpoint is rejected; no identical
continuation from update200 was started. Latest mixed-reset unit suite: 10 pass.

Broad unseen-motion, live stream, estimator and physical-mode handoff qualification
remain outstanding. No readiness, arbitrary-dance parity, or hardware authority
is claimed.

## 2026-09-07: joint-removal hypothesis tested; partial compatibility fix, not deployment

The network already removes the six absent output rows. This turn tested the
**untouched row-trimmed release**, not the earlier two-update generalist mislabeled
as initialization. No new weight training, robot command, deployment change,
commit or push was performed. Existing dirty gantry changes were preserved.

Two explicit simulation-only adapters now exist:

- `g1_true23_virtual_source_reference.py`: embed native23 reference angles in
  original29 kinematics with the six missing axes at zero, then compute the VR
  reference points expected by the frozen source encoder. This corrects the
  missing wrist-link translations/reference-frame mismatch. The integrated
  robot, measured observations and tracking ground truth remain native23.
- `g1_true23_release_action_diagnostic.py`: compensate the native tanh transform
  to recover bounded linear SONIC target semantics. The existing transform,
  raw-action bound, joint margins and effort limits remain active. Requests
  outside its reachable envelope are projected and reported, not silently
  treated as exact source actions. Nominal zero-training run projects 2.39% of
  joint requests, maximum 1.101 rad; this incompatibility remains significant.

The geometry-only correction changed initial-standing elbow means from about
`-0.016/-0.050` rad to `0.631/0.589`, against the same `0.600/0.600` reference.
This isolates a real cause of the arm bias. Simply replacing past lower-body
frames with the released future preview did **not** fix tracking; it worsened
the zero-training nominal full-lifecycle root p95 from 0.515 to 1.155 m.
Future-preview experiments are explicitly offline and not live-compatible.

All comparisons use the same full V5 planned-endpoint lifecycle: 1,841 controls,
including all 1,091 adapted dance frames, unchanged starts, model, gains and
thresholds. Every new run integrated the full duration, but **every dance
fidelity screen still fails**. Completing the duration is not successful dancing.

With both compatibility corrections, zero-training nominal dance p95 errors
improve: head `0.531 -> 0.287` m, hands `0.596/0.557 -> 0.351/0.321` m,
ankles `0.658/0.614 -> 0.371/0.393` m. Final standing maximum joint error
improves `0.667 -> 0.247` rad. However, identical X/Y push tests regress:
full-lifecycle root p95 changes `0.452/0.558 -> 1.032/0.837` m. The nominal
improvement is not a robust replacement policy.

The existing root100 weights were also evaluated with the adapters, without
changing weights. Root p95 changes from `0.655/0.969/0.471` to
`0.485/0.593/0.563` m for nominal/X/Y: mixed results, all fidelity screens fail.
Do not substitute this adapter into old deployment manifests or treat changed
encoder/action conventions as an exact resume of the old training lineage.

Evidence: `artifacts/g1_true23_generalist/release_semantics_20260908_v1/` through
`v6/` (directory labels retained; actual work date above). Each contains actual
physics traces, consumed encoder inputs and per-case reports. Used source
versions are preserved as `.snapshot.py` provenance files. The hash-bound
paired comparison is
`artifacts/g1_true23_generalist/release_semantics_comparison_20260907_v1/comparison.json`.
The comparison checks identical reference/model/initial state/gains/denominators
and exposes regressions rather than selecting only the nominal result.

Reproducer: `gear_sonic/scripts/diagnose_g1_true23_release_semantics.py`, with
`--modes causal_past_virtual_source --action-convention released_bounded_linear
--cases nominal standing_push_x standing_push_y`; provide asset root, motion
and a new output directory. Omit `--root-feedback-manifest` for exact zero-step
row-trimmed release; supply the existing root100 manifest only for the separately
labeled trained-policy comparison. **Simulation diagnostic, not hardware launcher.**

Next substantive work: integrate a separately versioned, training/CPU-consistent
reference/action convention and retrain/test whole-body tracking and disturbance
recovery on native23. Merely deleting outputs or continuing the mismatched old
training recipe is insufficient. Broad-motion generalization, live state
estimation, lifecycle fidelity and deployment remain unqualified. No universal
29-to-23 dance guarantee is established. Ninety-three focused tests passed;
these verify implementation, not dance performance.

## 2026-09-07: existing BONES-SEED recovered; evaluated continuation rejected

Correction to the earlier corpus-access claim: the dataset was already in
Ubuntu-22.04, documented in `G1_TRUE23_TELEOP_HANDOFF.md` and the original
`23dofsonic` work. No new download or license acceptance was needed to locate it.

- Archive: `/root/bones_seed/g1.tar.gz`, 23,499,973,647 bytes. Independently
  recomputed SHA256 matches the pinned release:
  `52580ea8bced72ea9e2ff1e7b68f01c51c7f1099581e9a46b7c87e1dec106d8a`.
- Existing CSV root: `/root/bones_seed/g1_extracted/g1/csv`.
  Metadata v004 contains 142,220 rows: 71,132 originals and 71,088 mirrors.
  Current filesystem has 98,854 nonempty files, 1,363 empty placeholders and
  42,003 missing files. Nonempty does not itself establish a complete CSV.
- Existing 5,318-clip native23 corpus remains at
  `/root/bones_seed/corpus/g1_true23_corpus_v3.npz`; it is not substituted for
  the new source-fidelity/feasibility-qualified training corpus.
- New index groups capture date, day part, actor and original take name before
  the 80/10/10 family split. All mirrors inherit the original's split. There
  are 550 test-split dance capture groups, of which 423 have nonempty original
  CSVs. These are available candidates, not 423 qualified dances.
- Fixed training diagnostic cohort: three originals each from dance,
  locomotion, standing and transition families. All 12 complete source CSVs
  converted successfully: 9,574 original samples at 120 Hz, all 29 named
  joints, original root trajectories and endpoints preserved. No test or
  validation clips used. No joint deletion, clipping or root reanchoring.
- A complete single-pass archive audit inspected 142,346 entries, recomputed
  the compressed archive hash and verified all 12 selected CSVs byte-for-byte
  against their archive members and conversion receipts. It extracted nothing.

Local-only source evidence is under
`artifacts/g1_true23_generalist/bones_seed_local_20260907_v1/`: metadata index,
frozen cohort, source conversions, archive-member audit and retarget reports.
The entire dataset-derived evidence directory is Git-ignored; raw motion and
metadata are not published. Motion Data by [Bones Studio](https://bones.studio/).
Underlying use remains subject to the
[BONES-SEED license](https://bones.studio/info/seed-license); this indexing work
does not independently establish license eligibility.

The first two unrestricted-excursion 2x-duration dance fits fail the unchanged
protected-task gates. A generic-source bug was fixed: hard refinement previously
accepted only the old happy dance's exact 2x/0.9 candidate. It now reconstructs
the declared bounded candidate and verifies the exact source-time map, including
120-to-50 Hz duration quantization, without changing any task/trajectory limit.
The two complete hard-refinement probes still report infeasible linearized
subproblems; this is not proof of physical impossibility. They remain rejected,
not accepted training data. Optional intermediate feasibility recovery was also
tested on both: worst normalized constraint excess fell from 5.225 to 0.724 and
12.737 to 0.448, but original protected-frame failures remain 493/664 and
266/366. Both remain rejected. Intermediate task envelopes are explicitly
diagnostic, with unchanged final acceptance and joint/root/temporal bounds.
The third full dance fit also fails the protected gates. No rejected source has
been promoted into controller training.

The first full standing-transition probe passes the existing 50 Hz grid gate:
890 original samples become 741 reference samples, actual duration scale
1.9977502812, no excursion reduction, zero protected-frame failures. Head/hand
p95 errors are 1.70/7.33/7.24 cm and maximum foot errors are 0.53/1.51 mm.
However, the added all-original-timestamp FK audit evaluates all 890 source
timestamps and finds two right-foot orientation regressions, at source indices
489/490. Maximum excess above the unchanged regression budget is 0.00020234 rad.
This clip is therefore **control-grid accepted only, not qualified training data**.
The previous grid-only report is preserved, not rewritten to hide this failure.
The v2 timestamp audit uses the existing serialized-joint tolerance of 1e-7 rad;
the earlier diagnostic's joint-bound flags were only 2.83e-9 rad of float32
rounding. Foot/task tolerances were not changed. No cached achieved task points
are used: original/direct/native23 FK is recomputed from saved source and target
joint/root data. This additional audit assumes linear joint/translation and
root SLERP interpolation; it does not prove continuous-time or dynamic safety.

Controller continuation is implemented with a new evaluated campaign boundary:
each local block preserves actor, critic, optimizer and counters but uses fresh
simulator/RNG state; each block is limited to 100 updates and requires the
previous full three-case CPU evaluation. Independent real-checkpoint audit
proved exact parent-state transfer at update 100, finite training state, unchanged
encoder and changes to all decoder/feedback tensors by update 200.

The update-200 candidate completed all 1,841 controls in all three cases, but
root p95 worsened to 1.095/1.021/0.778 m and nominal final standing joint error
to 0.761 rad, with final root speed 0.373 m/s. It is rejected for promotion.
The earlier evaluated checkpoint and all failures remain intact; identical PPO
was not blindly continued. Evidence:
`root_feedback_campaign_20260907_v1/segment_0100_0200/` under the generalist
artifact directory. Simulator qualification, broad-corpus training, live teleop
and hardware readiness remain incomplete. No robot commands were issued.

Verification checkpoint: 508 scoped tests passed (two ONNX deprecation warnings).
A subsequent opt-in `root_and_posture_v1` objective adds a dense error on actual
native23 joint position against the current received q10 reference, distinct
from the inherited requested-PD-target penalty. Its objective transition is
explicitly declared in a fresh continuation lineage; the legacy profile is
numerically unchanged. Forty-eight focused objective/environment/launcher/
campaign tests passed before starting a separate 100-update experiment from
the earlier root100 parent. That experiment is now complete, exported and fully
evaluated. All three cases complete 1,841 controls but root p95 is
0.884/1.047/1.336 m, worse than the parent in every case. Nominal final joint
error is 0.671 rad and speed 0.01297 m/s: lower speed does not compensate for
poor posture and dance fidelity. Nominal dance landmark p95 spans 0.730–0.855 m.
The posture candidate is also rejected; no automatic promotion or hardware run.
Independent real-checkpoint audit proves exact actor/critic/optimizer/counter
transfer, all ten encoder tensors unchanged and all decoder/feedback tensors
updated. Decoder ONNX probe max error is 1.90735e-6; encoder probes are exact.
Evidence is under `root_feedback_posture_20260907_v1/`, including the matched
full CPU comparison. Its checkpoint SHA256 is
`69ec9b9888d632c837c5cfb12057992d38f53b48cd4b6dbacfc34216fb7caa3a`.

Twenty synthetic all-original-timestamp audit tests pass, including lost
between-grid source poses, changed source-time maps, named-axis mismatches,
nonfinite arrays and attempted threshold changes. Final combined verification:
**534 tests passed**, zero failures or skips, two ONNX deprecation warnings,
272.48 seconds. Scoped Python E/F checks and `git diff --check` pass. Active
gantry source/header hashes remain unchanged. Earlier counts above are historical
checkpoints, not policy passes. Raw BONES motions and metadata remain Git-ignored.

## 2026-09-07: full root-feedback regression evaluated, not deployment-ready

The second GPU experiment completed 100 PPO updates / 51,200 environment
transitions. All 18 decoder tensors and all 36,864 feedback weights changed;
all 10 frozen encoder tensors remained exact. Four training episodes reached
the reference end, versus zero in the first experiment. This is episode
coverage, not tracking qualification. All 104 declared training source-material
entries still match their recorded bytes, including 92 local Python entries.
Export has exact encoder probes and decoder maximum absolute error 1.0133e-6.

Independent CPU evaluation uses identical initial state, full reference,
compiled native23 model, gains and effort limits across these policies:

| Policy | Completed controls: nominal / X / Y | Root position p95 m: nominal / X / Y |
|---|---|---|
| Preserved parent, zero feedback | 1841 / 1841 / 1841 | 0.502 / 0.431 / 0.842 |
| First uniform-rate root100, rejected | 1077 / 981 / 829 | 3.741 / 2.709 / 1.705 |
| Second feedback-priority root100, experimental | 1841 / 1841 / 1841 | 0.655 / 0.969 / 0.471 |

The stopped first candidate has different completed denominators; its p95
numbers are not used for percentage improvement comparisons. The second
candidate completes all 1,091 dance samples in all three cases, without pose
resets, fallback controllers or height/tilt stops. Nominal source landmark
errors improve 7.6–20.5% over the parent, but the X-push source errors worsen
21.5–82.7%; Y results are mixed. Neither parent nor candidate passes fidelity.
Final nominal standing joint error is 0.681 rad (parent 0.645 rad), and root
position error is 0.654 m. No candidate is promoted; no hardware was run.
The pinned original23 baseline stopped at 645/1841 controls during the dance,
before either return reference applies; its result is retained separately.

On identical saved parent observations, dance p95 decoder action drift drops
from 0.473 to 0.069 RMS while feedback effect rises from 0.00100 to 0.01623.
The new optimizer profile addresses the diagnosed imbalance, but does not
establish an overall better controller or normal-standing return.

Evidence: `root_feedback_regression_20260907_v2/comparison.json`, update/export
receipts and shared-observation audit; complete CPU results in
`planned_v5_endpoint_priority100_20260907_v1/report.json`. All are under
`artifacts/g1_true23_generalist/`. Candidate checkpoint SHA256:
`f8c4f2611aab6b09b69973b7d65f5f28869deead99f89e99290cb7178d5d8421`.

Verification: 448 tests pass across all 24 generalist/root-feedback/native
actuation test modules, including legacy and differential-rate resume. Ruff
E/F checks pass across all 56 scoped Python files. The two warnings are ONNX
export deprecations. Final JUnit receipt:
`root_feedback_verification_20260907_v1/pytest_final.xml`. Hash-bound staged
source/evidence bytes match their on-disk inputs; existing dirty gantry files
remain outside the commit. Checkpoints, ONNX binaries and NPZ traces stay local.

Next controller work must improve tracking and standing together, with the
existing completed parent as a regression baseline. Do not respond to these
failures by retrying hardware, easing fidelity gates or changing mode-transition
code. Broad licensed training/held-out data, dynamic/contact qualification,
causal retarget deadlines, interrupted teleop and a qualified physical root
estimator are still outstanding. The deployment goal is not complete.

## 2026-09-07: endpoint-return correction and rejected first training candidate

**Reference correction:** source choreography ends near [8.0836, -0.0848] m.
The old generated return asked it to travel approximately 8 m back to configured
origin in 2 s (peak reference speed7.50 m/s), while interpolating into standing.
The prior whole-lifecycle 8 m error was therefore mostly this invalid/unqualified
return request, not 8 m of dance drift. The independent XY-observability result
still holds, but it must not be used to misattribute this return-reference bug.

New endpoint lifecycle v2 preserves every source sample and all1841 controls,
but requests normal standing at the **fixed planned terminal XY and heading**.
It does not chase measured robot position or change source-world tracking gates.
Horizontal return-reference displacement/speed are exactly zero. Root height
and joint posture blend into configured standing. Contact/force feasibility
and endpoint-velocity matching remain explicitly unqualified. Legacy
`configured_origin` generation stays available and bit-identical for historical
reproduction; new root training/evaluation defaults to `planned_endpoint`.

Corrected-timeline CPU comparison (nominal, +X and +Y 4 Ns standing impulses):

- Untrained root branch / preserved parent:1841/1841 controls in all3 cases,
  but source fidelity fails; whole root position p95 is0.50/0.43/0.84 m.
  Final standing joint error is about0.64 rad, so survival is not normal standing.
- First root100 candidate (51,200 training transitions):1077/981/829 controls,
  all stopped by absolute height/tilt before finishing dance. Rejected, not
  promoted. Its optimizer/critic/loss values are finite and ONNX parity passes;
  those facts do not establish motion quality.
- Original pinned23dof policy on matched source/initial state/model:645/1841
  controls, failing during dance before either return reference is used.

Shared-observation diagnosis separates decoder drift from added feedback: dance
p95 action RMS change from decoder weights0.473, feedback branch effect0.00100.
Acquisition figures are0.128 versus0.000040. First experiment changed the base
controller far more than it trained the missing feedback.

Second bounded experiment starts from the same surviving old parent. New
`feedback_priority` profile uses base/decoder learning rate5e-7, root conditioner
1e-4, exploration5e-7 and critic3e-4, with fixed schedule and strict versioned
group-rate validation. All decoder layers remain trainable; encoder remains
frozen. Saved every20 updates, evaluated independently by100 updates. No physical
limits, source-frame denominator or acceptance thresholds are relaxed.

Evidence directories: `planned_v5_endpoint_root0_20260907_v1`,
`planned_v5_endpoint_root100_20260907_v1`, `root_feedback_regression_20260907_v1`
(including shared-observation drift audit), and second run
`root_feedback_regression_20260907_v2`. Deployment goal remains incomplete.

## 2026-09-07: root-conditioned controller and accepted planned reference

Deployment-readiness work resumed in simulation only. No hardware commands or
changes to existing dirty gantry/deployment files. Physical dance and live teleop
remain unqualified; the existing goal is not complete.

**Concrete reference milestone:** the full planned 29-DoF happy dance now has an
accepted native23 adaptation. All 546 original frames are preserved through an
explicit 2x time map into 1,091 frames. Task-space adaptation is 10.04%, below the
approved 20% ceiling. Explicit second-order-cone constraints repair foot
orientation and COM regression without changing any final acceptance gate.
All protected failure categories are zero. Independent cold-file FK and raw
planned-source lineage checks pass. Head p95 error is 3.76 cm; hands 6.85/7.12 cm;
maximum foot position errors 1.81/2.62 mm. This is kinematic reference acceptance,
not dynamic feasibility or controller qualification.

Evidence: `artifacts/g1_true23_generalist/planned_dance_retarget_20260907_v5/`.
Accepted motion SHA256:
`dd325625b507d4815cae2f2795b1ae38c7f2ec33731f0c7ecd9544f26f005a91`.
V4 forensics now resolve all 839 unique invalid frames into overlapping COM170,
left-foot orientation267 and right-foot orientation512 failures; prior rejects
remain on disk, never relabelled accepted.

**Architecture fix implemented:** a separate 9-value input carries root position
error and desired/measured linear velocity in measured pelvis-yaw coordinates.
The frozen SONIC267 encoder/64-token branch and original decoder994 input remain
unchanged. A zero-initialized 9x4096 projection adds to the decoder's first
preactivation; all decoder tensors plus that branch, bounded noise and critic
train. New checkpoint and two-input ONNX contracts reject legacy mislabelling.
Zero initialization preserves the old parent exactly, including with nonzero
feedback. Simulation ground truth is explicitly not a qualified hardware pose/
velocity estimator.

The new task binds tracking rewards and terminations to held received q10
fixed-world targets; causal velocities use q9-to-q10 differences. Original q9
tokenizer semantics are preserved. Root error cannot be hidden by reanchoring
the reference to measured robot XY. Reward timing remains post-physics before
command advance, including MJLab's existing 2 ms stale derived reward state;
actor observations are refreshed current state.

`train_g1_true23_root_feedback` provides bounded smoke, local regression and
audited train modes. Regression is explicitly unaudited/non-generalizing, not
an enlarged smoke run or a bypass of corpus ownership. Each session stops at
100 updates for independent CPU evaluation. The accepted planned reference is
the selected lifecycle input; training/evaluation outcomes follow below.

## 2026-09-07: simulation-first native23 generalist implementation

**New architecture blocker:** with reference fixed, shifting robot XY by
[8, -3] m produces bit-identical actual v2 encoder267, history930, decoder994
and raw23 action. Actual training features have the same invariance. The fixed
interface sees no root-position error and reanchors reference XY to the robot.
It cannot recognize a persistent source-world position offset as an error.
More training can improve relative motion but cannot supply missing feedback.
Source-world path/return requires a separately versioned root-feedback input,
or explicit narrowing to root-relative qualification. This contract decision
is not silently made and world-frame acceptance gates are not relaxed.

**Physical dance and live teleop remain unqualified. Simulation work does not
require robot access.** All work below is local CPU/GPU simulation, reference
retargeting, training or artifact validation. No DDS, SSH, robot mode change or
motor publishing occurs. The original `23dofsonic` checkout and all pre-existing
dirty hardware files remain untouched.

New generalist path freezes the released SONIC encoder/FSQ and trains all 18
native23 decoder tensors (37,390,871 parameters), critic and bounded exploration.
The 267/64/994 SONIC interfaces remain explicit; six missing axes are observation
padding, never physical motors. Simulation separates nominal native-model full
effort saturation at 500 Hz from the existing quarter-effort/slew gantry setup.

Two real four-environment/two-update GPU smoke runs complete. Every decoder
tensor changes and the encoder stays bit-identical. Smoke v2 adds recursive
61-file source dependency binding; exact ONNX encoder probes and decoder maximum
absolute error 2.115965e-6 pass. These are initialization/training/export checks,
not a trained general dance controller. V1 and v2 decoder weights differ, so
their measured dynamics receipts are not interchangeable.

Fresh v2 CPU lifecycle also completes all 1296 controls and all 546 source
frames, but source task-point p95 remains 1.600–1.685 m. Final root is
[7.90178, 1.39630, 0.75308] m, root speed peaks at 0.02705 m/s during standing
proof and standing joint error reaches 0.6314 rad. V2 fails fidelity too.

The common nominal CPU benchmark and complete standing/dance/standing lifecycle
run one actor with no mid-motion pose reset, history reset, fallback or alternate
standing controller. On v1, the new generalist integrates all 1296 lifecycle
controls and settles upright, but drifts about 8 m and misses requested arm
posture. It fails dance fidelity. Original walk v14 rejects raw action after
352 lifecycle controls; prior lifecycle LoRA integrates 1296 but also fails.
Duration completion must not be reported as successful dancing.

Two earlier assumptions are corrected with actual evidence:

- Old native23 dance targets tracked the **recorded29 policy rollout**, not the
  original requested choreography. Planned-source discrepancy is approximately
  0.57–0.64 m at task-point p95. New adapter uses only `planned_qpos50`.
- Exact original walk replay depends on its old observation phase. Its angular
  velocity reads stale pre-final-substep `cvel`; the current controller refreshes
  kinematics. First control matches exactly, then the second raw action differs
  by 0.127677. Matching gains alone does not reproduce historical frontend
  semantics. This is a simulation discrepancy, not a physical fault diagnosis.

The full planned 546-frame dance now reaches the task-space solver for all 12
bounded tempo/excursion candidates. Fixed-root v3 rejects all candidates: at
2× duration/90% excursion, feet improve below 1 mm p95 but hands remain
17.49/18.74 cm and head 15.13 cm. Original source-limit excess is retained and
reported; target limits and final gates are not weakened. No failed reference
is emitted as usable motion.

One bounded v4 root-orientation+23 refinement fixes the structural head-motion
limitation of fixed-root IK. At 2× duration/90% excursion, all 1091 frames solve;
head p95 improves to 3.21 cm, hands to 5.84/6.12 cm and feet stay below 0.75 mm.
ROM, 4.975-rad/s speed, 79.601-rad/s² acceleration and serialized root bounds
pass. It still rejects: 252/1091 frames pass the unchanged protected mask.
Weighted optimization trades some foot orientation against upper-body fit;
the aggregate report does not localize all 839 failed frames by criterion.
This is a failed constrained solver result, not a physical impossibility proof.
No additional run or relaxed threshold follows. Report SHA256:
`0f32fe59e30001945479a10e9d61e996577fb1bab576a4b839131877082d3657`.

Corpus audit splits original recording families 80/10/10 before augmentation and
binds license/lineage evidence, named joints, timing and file hashes. Acceptance
requires an external immutable full-source phase plan, at least 100 independent
held-out dances, three seeds and 95% complete lifecycle success. Missing/rejected
cases remain failures; a one-frame or cropped self-declared dance cannot pass.
Local 127 candidate files are not 127 independent recordings. The earlier claim
that broad corpus access was missing was incorrect: the existing WSL BONES-SEED
archive and metadata were subsequently located and verified; see the latest
entry. No new gated dataset license was accepted.

Nominal acquisition and lifecycle training modes now derive references from
validated original train assets. Timed command resampling is bypassed: inherited
state writes occur only inside environment reset. Explicit parent initialization
transfers trained actor/noise into a fresh critic/optimizer/counters and new
lineage; exact same-stage resume is separate. V3 GPU smoke verifies parent v2
actor equality for 29/29 tensors, then all 18 decoder tensors update while all
ten encoder tensors stay fixed. All 86 bound local Python sources match.
Only initial standing is exercised: 4 environments, 16 controls each, 64 total
transitions, 2 updates, 4 initial resets, zero completed reference timelines.
This is not completed lifecycle training, domain randomization or teleop proof.

Final combined verification: **315 tests pass, zero failures or skips**, in
135.39 seconds; Ruff E/F and format checks pass for all 37 new Python files.
Earlier 274-test evidence remains retained as a historical snapshot. Causal per-frame
retargeting has no lookahead and rejects stale input without manufacturing an
action, but its 100-frame moving test misses 40 deadlines at 20 ms. This is not
PICO integration or real-time qualification. Full generalization, the remaining
randomized/interrupted-input curriculum, constrained planned-reference
feasibility, contact/fall evidence and the live reference path still require work.

Implementation, exact evidence labels, runnable entry points and remaining
qualification are documented in `G1_TRUE23_GENERALIST_SIM.md`. Evidence root:
`artifacts/g1_true23_generalist/`. Binary motions, checkpoints and measured-state
videos remain local; diagnostic reports do not authorize hardware use.
Final verification receipt:
`artifacts/g1_true23_generalist/verification_20260907_v2/verification.json`,
SHA256 `d7159d0c482b2c8249853a13f13d79b08296372767c9b96d7351bc79e19c0896`.
It binds 37 stable Python sources, 66 evidence files, final JUnit and unchanged
active C++ source/header. New evidence/source Git line endings are pinned to LF;
existing hardware edits remain outside the simulation commit.

## 2026-09-07: read-only incident capture tested at 1,000 callbacks/s

**Still NOT ready for physical dance or live full-body teleop.** This turn
finishes a diagnostic prerequisite, not another policy experiment or hardware
handoff change. Robot Ethernet remains disconnected. No robot, DDS participant,
SSH, publisher, mode request or training run is started. The pre-existing
hardware edits remain unchanged and unstaged; the active C++ source/header
still match the prior handoff audit's SHA256 values.

New passive recorder subscribes to HG LowState, raw/user LowCmd and sport /
motion-switcher RPC request/response topics. It preserves immutable reserialized
CDR, receipt order/time, source/IDL/native-CRC hashes and explicit collector
loss/error counts. It neither publishes commands nor asks for robot mode.
Original wire bytes and losses upstream of its callback are not observable.

An initial inline decoding design drops **4,365/10,000** paced synthetic
callbacks; its failure evidence is retained. The corrected design writes raw
records first, then interprets the saved file in a separate offline command.
CRC-valid native23 status edges, including a one-sample raw bit-30 pulse, are
preserved without labeling them a physical diagnosis. CRC/tick discontinuities
break the edge baseline; absent axes never become controlled motors.

**73 tests pass, zero failures or skips** with real SDK IDL/native CRC and
fake subscriptions; network/DDS constructors are forbidden in those transport
tests. The initial SDK default-factory test errors were resolved by using the
SDK's actual IDL constructors, without changing the SDK. Ruff E/F and format
checks pass for all seven new Python files.

The final **30-second / 30,000-callback** raw-first benchmark runs at requested
500 LowState + 500 command packets/s, with **zero collector drops**, both
synthetic bit-30 edges recovered and queue peak **335/2,048**. Callback p99 is
**0.716 ms**, maximum **2.082 ms**; offline decoding takes **27.066 s** after
capture. This does not exercise DDS or qualify real-time scheduling. A prior
10-second run also preserves all 10,000 packets but has a **116.677 ms** host
pause, so zero file-queue loss must not be sold as live transport reliability.

Evidence root:
`artifacts/g1_true23_frozen_lora/incident_capture_20260907_v1/`.
`verification.json` binds the completed 73-test JUnit record, benchmark reports,
diagnostic source and unchanged active hardware source/header. SHA256:
`d56c00dd9f76cfb9f4f8bab263621105b415f311d2c1e7be662fa1c03c7f0d18`.
Raw synthetic captures are local artifacts, not physical incident evidence.
Usage and limits are in `G1_TRUE23_INCIDENT_CAPTURE.md`.

**Deployment path remains:** establish the actual firmware and supported
full-body ownership protocol first; separate normal standing handback from
fault/damp recovery and qualify its failure cases; then qualify the complete
native23 dance plus standing return and live PICO stream. The candidate
`rt/user_lowcmd` retained-service protocol is not yet validated for this robot.
Policy remains **2.06/10.7 s** of historical-start dance, **0/250** return
controls. No new readiness claim or motion authorization is made. Next
hardware-side prerequisite is reconnecting the observer and obtaining current
firmware/telemetry, read-only; do not launch another dance to manufacture a fault.

## 2026-09-07: actual handoff RPC failure reproduced; qualification corrected

**Still NOT ready for physical dance or live full-body teleop.** This turn
targets the deployment handoff rather than starting another training run.
No robot/DDS/SSH/motor/mode operation occurs. The local robot Ethernet adapter
reports disconnected; no current firmware or physical robot state is inferred.
All pre-existing hardware edits remain unchanged and unstaged.

New offline fixture compiles the current runtime's actual restore functions,
constants and evidence gate with only RPC, LowState and time substituted.
It reproduces a concrete path: accepted damp FSM 1, rejected stand FSM 4,
repeat, unsuccessful exit with the mock still damped. Successful mock recovery
also has a 6.5 s gap between this routine's health reads. The DDS monitor thread
is not executed, so this is not a measured subscriber outage. Current final
motor-health checks reject unhealthy telemetry, but can accept FSM 801 with
both knees at 1.5 rad. None of these injected outcomes establishes the cause
of the real bit-30 motors-off latch or a particular historical incident.

`qualify_g1_true23_active_lifecycle_no_robot.py` now executes the RPC audit.
On the same current source/binaries, its old version returns success; its
updated schema-2 version returns **failure / exit 2** while the original core
and binary/source checks still pass. Zero damping LowCmd frames cannot hide
high-level damp requests. The new failure reasons also include stand commands
after an injected disable and accepting crouched telemetry as normal standing.
This fixes qualification coverage, **not the hardware handoff itself**, and
does not replace or automatically disable the launcher's existing gates.

**26 tests pass against each of current worktree and exact `64f2bc3` committed
source/header snapshots.** Committed snapshots match their Git blob IDs;
Ruff E/F and formatting checks pass. The old committed restore routine's
missing physical gate and failure to recover from inert FSM 0 remain explicit.
The tests characterize defects; they are not 52 physical-motion successes.
Completed verification evidence is under
`artifacts/g1_true23_frozen_lora/restore_rpc_20260907_v1/validated/`;
`verification.json` SHA256:
`c77e90a6bda8d6e3924da5d496c40b0543901ae60feee58fef7a81c481f8b3b1`.

Found a materially different official **full-body** ownership protocol:
[Unitree's pinned example](https://github.com/unitreerobotics/unitree_sdk2/blob/30405b31d82f137d48f33cbba095d149749db601/example/g1/high_level/g1_userctrl_dds_example.cpp)
uses `rt/user_lowcmd`, `SwitchToUserCtrl()` and internal LAST handback. Current
runtime uses service release plus `rt/lowcmd` and never enters that protocol.
The example starts in **passive FSM 1**, so it is not proof of seamless
standing-to-standing transfer and must not be copied into a robot run.
Firmware support, native23 command semantics and ownership acknowledgment
remain to be established before transport integration. Full details, exact
reproduction evidence and the next integration boundary are in
`G1_TRUE23_HANDOFF_FINDINGS.md`. Policy dance remains 2.06/10.7 s with no full
standing return; transport work does not change that result or qualify PICO.

## 2026-09-07: 100 lifecycle updates improve prefixes, but no full motion passes

**Still NOT ready for physical dance or live full-body teleop.** The single
planned 100-update CPU lifecycle run completes successfully in **3,095.82 s**:
**38,509 actual active-policy actions / 1,416 optimizer minibatches / 800
full-request training attempts**. The two-update smoke was separate, not
counted as resumed training. Initial actor, source motion, gains, effort/slew
limits, frozen encoder/base decoder/action std and standing retention remain
as documented below. No training source or controller code changed during
this run. All original requests remain in the evaluation, including the
unavailable elbow and separate standing prerequisites.

| Complete request | Original v14 100 | Prior LoRA 100 | Lifecycle +100 | Requested controls |
|---|---:|---:|---:|---:|
| Hand crawling | 16 | 64 | 127 | 595 |
| Dance, reference start | 16 | 64 | 96 | 535 |
| Dance, historical start | 24 | 60 | 103 | 535 |
| PICO upright | 17 | 37 | 42 | 1,013 |
| PICO standing | 7 | 37 | 41 | 1,013 |
| PICO crouch | 20 | 25 | 25 | 1,013 |
| PICO walk 001 | 16 | 22 | 22 | 684 |
| PICO walk 010 | 11 | 24 | 24 | 499 |
| Synthetic standing, reference start | 30 | 500 | 500 | 500 |
| Synthetic standing, acquired | 36 | 500 | 500 | 500 |

Historical-start dance improves from **1.2 s to 2.06 s of 10.7 s**; its
standing return is still **0/250 controls**. Reference-start dance reaches
1.92 s. Hand crawling reaches 2.54 s of 11.9 s. All available complete motion
evaluations still fail with `TargetIntersectionError`. Stationary standing
retains **250/500/250** acquisition/active/return. The historical-dance failure
is a left-ankle-pitch effort/position/slew intersection gap; it does not explain
the physical motors-off incident and does not authorize increasing a limit.

This is a real improvement in these particular simulator evaluations, not
full-dance completion, a multi-seed reliability estimate, an equal-total-budget
comparison, or proof that the training-engine change alone caused it. The
new actor remains unqualified; no deployment export or robot operation occurs.
No additional training run is started by the completion checker.

Independent comparison verifies actual action counts across all 800 training
attempts, checkpoint/optimizer counters, changed actor and critic, unchanged
action std, identical requests/limits, and **20 new evaluation traces / 43,185
physics steps** with continuous state/time, matching PD effort and intact
effort caps. Existing **704 focused tests** remain tied to the unchanged
source/test hashes; they are not new hardware evidence. Detailed outcomes and
failure joints are in `completed_100_comparison.json` under
`artifacts/g1_true23_frozen_lora/lifecycle_ppo_20260907_v1/`.
Comparison SHA256: `9133803919d0694fd068192f3b49e8ec97bd65cb65bd316d53fa7b93b8b56dd4`.
Experiment SHA256: `44aae49d873b16a0eb0074477e0ff77368731fad3669b17c3d20cc30defdaa55`.

The saved-state attenuation check below motivates investigating how much
sampled action variation produces identical complete 20 ms control sequences.
[Clipped Action Policy Gradient](https://proceedings.mlr.press/v80/fujita18a.html)
is relevant published work on reducing estimator variance from action clipping.
It is **not** a ready-made proof for this changing 2 ms projection: any use here
must preserve the full held-action sequence, rewards and actual target-history
feedback. No clipping-aware estimator or altered exploration scheme has been
implemented or tested in this run. Longer training would also need a deliberate
runtime/storage budget; this run's full traces remain preserved.

## 2026-09-07: full-lifecycle CPU PPO runs; two-update smoke is not a dance fix

**Still NOT ready for physical dance or live full-body teleop.** The new
experimental trainer now executes real PPO in the same native CPU MuJoCo
implementation used by the complete-request acceptance tests. Two startup
bugs were fixed: the checked actor loader requires `expected_contract`, and
standing-retention metadata is a descriptor, not a path-to-hash map. Both
failed attempts stopped before simulation or learning. Their logs and exact
source snapshots remain preserved, not overwritten by the successful run.

The successful smoke performs **2 new updates / 656 actual active-policy
actions / 24 optimizer minibatches** across **16 complete-request attempts**.
These are attempts at the full clips, not 16 successfully completed motions.
Every attempt starts from the historical snapshot, performs 250 actual
standing-acquisition controls, requests the entire motion, then requests
250 standing-return controls. No state/history reset occurs at handoff.
Failures and the real partially executed/rejected terminal action are retained.
The fixed acquisition/return controller's actions never become PPO samples.
An incomplete motion always receives a negative terminal assessment, including
when an early return succeeds. No completion bonus rewards a shortened dance.

The actor starts exactly from prior IEEE LoRA 100, with a fresh critic/Adam.
Only decoder LoRA and the critic train; encoder/base decoder/action std remain
frozen. The same standing-output anchor, motor gains, effort caps, target slew,
267/930 observation boundaries and 50 Hz/500 Hz cadence are retained. The
original eight-request set, unavailable elbow and separate standing cases
remain explicit. The new method changes the simulation/training engine,
episode/reset sampling and rewards, and adds no sensor noise. It is not a
one-variable ablation or a reproduction of the original SONIC training recipe.
Checkpoints have a distinct CPU lifecycle schema; they are not relabelled as
MJLab resume checkpoints. No ONNX/deployment artifacts are emitted.

The initial Torch IEEE evaluation reproduces all prior ONNX completion counts
and standing outcomes; this does not claim bit-identical full trajectories.
After two updates, all full motions still fail. Stationary standing is retained.

| Full request | Original v14 100 | Prior LoRA 100 | Lifecycle +2 | Requested controls |
|---|---:|---:|---:|---:|
| Hand crawling | 16 | 64 | 66 | 595 |
| Dance, reference start | 16 | 64 | 65 | 535 |
| Dance, historical start | 24 | 60 | 60 | 535 |
| PICO upright | 17 | 37 | 37 | 1,013 |
| PICO standing | 7 | 37 | 35 | 1,013 |
| PICO crouch | 20 | 25 | 25 | 1,013 |
| PICO walk 001 | 16 | 22 | 22 | 684 |
| PICO walk 010 | 11 | 24 | 28 | 499 |
| Synthetic standing, reference start | 30 | 500 | 500 | 500 |
| Synthetic standing, acquired | 36 | 500 | 500 | 500 |

All three historical-dance returns complete **0/250** controls. Lifecycle +2
preserves stationary acquisition/active/return at **250/500/250**. It adds two
updates to the prior actor, so this is not an equal-total-budget comparison.
The smoke verifies implementation, not enough training or controller quality.
A single 100-new-update experiment was launched with unchanged full evaluation
before and after training, after the focused regression suite passed. It starts
from the same prior actor with fresh critic/Adam, not by claiming exact resume
from this smoke. Its completed results are recorded in the newer section above;
no deployment claim is made.

Independent checks verify **36 continuous traces / 93,247 physics steps**,
including all training attempts, before/after evaluations, exact initial
adapter identity, real optimizer counters, Gaussian action log probabilities,
rewards, state/time continuity, effort caps and active PD equations. The
full focused regression passes **704 tests in 187.94 s across 52 modules**,
including 24 new tests and the same two documented legacy asset-root
substitutions. This is not a whole-repository or hardware-test claim. Ruff E/F
and format checks pass for the four new source/test files.
Evidence: `artifacts/g1_true23_frozen_lora/lifecycle_ppo_20260907_v1/`.
Comparison SHA256: `dc5e33dce12c2465551afe6857f558927e705a1edcfc5dad1fc7f86a510268c7`.
Experiment SHA256: `76e6ec86f834b4b636771a48e9696a35d462e882021bc313c2c2bc9942d10514`.

No robot connection, DDS, SSH, arming/mode command, motor operation or deployment
change occurred. Existing dirty hardware work remains untouched. Physical
bit-30 motors-off causation and native Unitree FSM handoff remain unproven;
the simulator's standing controller is still a 29-to-23 compatibility actor.

While the 100-update run was running, a read-only recount of **76 archived
physical execution logs** finds four with a positive reported damping tail,
48 with zero and 24 without this field. None contains raw motor-status/mode
fields identifying the driver-disable edge. Terminal records include 16
effort faults and eight position faults; these software guards are not a
diagnosis of the later latched motor-disable incidents. The separate healthy
snapshot is from 2026-09-05 10:36:44–10:36:50 UTC, not from those fault edges.
It cannot rule out a transient electrical/thermal event or establish what
firmware bit 30 means. Earlier notes that exclude power/thermal causes from
post-event measurements are not substantiated. This narrow archive audit
does not claim no additional evidence exists elsewhere or establish current
robot state. See `historical_incident_evidence.json` in the experiment folder.

A same-recorded-state check of all eight training attempts at updates 1 and
50 reproduces the actual requests/projections bit-exactly, then substitutes
the saved policy mean at those same states. Across these 16 traces, **74–88%
of joint/substep requests are clipped**, and projected action-noise RMS is
**6–8%** of its pre-projection target RMS. For the dance specifically, the
ratios are **7.61% / 8.01%**; sampled and mean targets project identically on
**39.6% / 36.1%** of joint/substeps. This is substantial attenuation, not total
loss of policy influence. It is a possible learning bottleneck, not proof of
the training failure's cause, a closed-loop noise ablation, or justification
to relax limits. No extra physics/policy calls or parameterization changes
occurred. Evidence: `action_transmission.json` in the same experiment folder.

## 2026-09-06 continuation: reset curriculum rejected on full-motion tests

**Still NOT ready for physical dance or live full-body teleop.** The current
change addresses a measured training problem, not a proven physical-damping
cause. A real three-update smoke run passes. The requested 500-new-update
run **fails after 410 logged completed updates / 209,920 rollout transitions**:
a synthetic reset exceeds the configured 0.2 m floor-lift bound. No larger
lift is allowed. Last saved checkpoint is update 400, and the failed reset
state was not captured; its exact source phase and required lift are unknown.
Incomplete final-rollout transitions are not counted as completed PPO updates.
Both saved candidates fail every complete motion. Curriculum +100 preserves
stationary standing, but +400 loses it despite the output-retention penalty.
Neither candidate replaces the previous LoRA 100 policy.
No robot, DDS, SSH, mode/arming command, motor operation or deployment change.
Existing dirty hardware work remains untouched.

### Why this training change

Six no-learning probes use the current IEEE LoRA 100 actor, 32 environments
and 128 controls each: **24,576 transitions / 768 guarded actor calls**.
An independent recount checks actual termination masks and censored episodes.
The four action/sensor-noise combinations start with exactly equal physical
state, source phase, joints and previous target; all have a completed-episode
median of **3 controls**. Removing either or both noise sources alone does
not demonstrate an improvement. With both noises disabled, removing reset
perturbations raises the median to **25**, and correcting detected floor
overlap raises it to **30**. All six initial source phases match; later
adaptive resets diverge, so this is a single-seed diagnostic, not a paired
long-horizon quality estimate or a complete-motion pass.

New `train_g1_true23_reset_curriculum.py` starts from that checked current
actor, with fresh critic/Adam/counters and the existing standing-output
retention. It keeps the complete corpus, encoder, action noise and sensor noise,
motor limits, gains, cadence and rewards unchanged. Only reset pose/velocity/
joint disturbances are scheduled: zero through 1,600 actual environment
controls, linear ramp to the original amplitude at 6,400, then full amplitude.
Detected floor overlap is corrected at every reset by bounded root-z lifting;
all other positions and all velocities are preserved. This does not prove
self-collision freedom, equilibrium or physical reachability.

The real smoke performs **3 new updates / 96 transitions**, starts with the
exact prior LoRA 100 actor, uses empty fresh Adam state, changes actor and
critic, and records six full-amplitude reset rows with unchanged action std.
Resume is intentionally rejected; no partial-state resume claim is made.
The failed run targeted **500 new updates / 256,000 transitions**. Checkpoints
100 and 500 were fixed in advance for unchanged full-request evaluation,
including the complete 535-control dance and 250-control standing return.
Update 500 does not exist and remains explicitly not executed. Update 100 is
evaluated as planned; update 400 is an additional last-saved-checkpoint
diagnostic, not a replacement for the missing planned result or a successful
500-update run. Both correctly paired simulator exports pass their existing
export checks; neither is qualified or exported for hardware. Evaluation
preserves all 11 records, including the unavailable original elbow. All
original clip lengths, tempo, motor gains, slew, effort, histories and full
standing-return requests remain unchanged.

| Complete request | Original v14 100 | Prior LoRA 100 | Curriculum +100 | Curriculum +400 | Requested controls |
|---|---:|---:|---:|---:|---:|
| Hand crawling | 16 | 64 | 65 | 68 | 595 |
| Dance, reference start | 16 | 64 | 65 | 65 | 535 |
| Dance, historical start | 24 | 60 | 65 | 60 | 535 |
| PICO upright | 17 | 37 | 35 | 42 | 1,013 |
| PICO standing | 7 | 37 | 31 | 40 | 1,013 |
| PICO crouch | 20 | 25 | 26 | 26 | 1,013 |
| PICO walk 001 | 16 | 22 | 29 | 22 | 684 |
| PICO walk 010 | 11 | 24 | 13 | 29 | 499 |
| Synthetic standing, reference start | 30 | 500 | 500 | 73 | 500 |
| Synthetic standing, acquired | 36 | 500 | 500 | 317 | 500 |

Historical-dance standing return completes **0, 0, 2 and 0 of 250 controls**,
respectively. Curriculum +100 completes stationary acquisition/active/return
at **250/500/250**; +400 regresses to **250/317/0**. The best new historical
dance is only **1.3 s of 10.7 s**, followed by 0.04 s of the required 5 s return.
All failed motion cases stop on an empty effort/position/slew intersection.
The initial improvement on less-disturbed synthetic resets does not establish
durable closed-loop tracking or recovery. More identical updates or simply
raising the reset-lift bound is not justified by this experiment.

The curriculum starts from the already trained prior LoRA 100 actor; +100 and
+400 are **additional** updates. This is not an equal-total-training-budget
comparison with original v14 or a statistical estimate from multiple seeds.
The standing controller remains a 29-to-23 compatibility actor, not native
Unitree FSM transfer. Physical bit-30 motors-off causation remains unknown.

Evidence: `artifacts/g1_true23_frozen_lora/reset_curriculum_20260906_v1/`
(`smoke_verified.json`, `reset_lead_verified.json`, failed `train500.log`,
`interrupted_run.json`, `interrupted_export_evaluation_report.json`). The
original success-only continuation correctly does not run without a completed
experiment report. A separate interrupted-run
evaluation preserves that failure and evaluates the saved policies. No second
training run is started.

**680 tests pass in 186.90 s across 51 focused modules**, retaining the same
two documented legacy module asset-root substitutions. Ruff format and
critical-error checks pass for all six new source/test files. This is not a
whole-repository or hardware-test claim.

Independent `full_result_comparison.json` checks equal request identities,
lengths, gains, limits and timing against both prior implementations. It
verifies **20 new continuous traces / 33,325 actual physics steps**: adjacent
positions/velocities and engine times match exactly, generalized actuator
force equals recorded effort, the existing effort cap holds, active PD
equations agree within 1e-12 Nm, and no engine warnings occur. This verifies
the rejected trajectories, not successful full-motion control.

Comparison SHA256: `13105deaa25c56abb28607d17c487206a1e20a46beb7cb95e7498ff88f13b038`.
Interrupted evaluation SHA256: `3bdd4e0c6937a3b0cfc2ed176d04a211ee4c3bdd4aab3e8a7dea3105c6f2a6e1`.
The next training change must address sustained closed-loop motion and
standing recovery; this reset-only intervention is not a deployment fix.

## 2026-09-06 continuation: exact-prefix standing-return recovery windows

**Still NOT ready for physical dance or live full-body teleop.** Exhaustive
early-return tests now distinguish a feasible immediate command from an
actually successful five-second simulated return. LoRA 100 returns after
dance controls **1–14**, LoRA 300 after **1–13**, and corrected original-v14
100 after **1–2**. Every later tested boundary fails. These are observations
on three particular simulator trajectories, **not safe live cutoffs**, proof
that another recovery controller cannot work, or shortened-dance success.
The original complete 535-control dance request remains failed for all three.

No robot connection, DDS, SSH, mode/arming command, motor operation, deployment
change or training occurred. Existing dirty hardware work remains untouched.
The physical bit-30 motors-off cause is still unknown; these simulator results
do not establish why the real robot damped.

### Re-executed trajectories, not reconstructed handoff states

New `audit_g1_true23_recovery_window.py` binds each existing full-request
report and reloads its correctly paired policy. Each candidate first repeats
the entire failed historical-start lifecycle: every result field and saved
array must exactly equal the original full run. It then tests **every**
completed 50 Hz dance boundary, each time re-executing the unchanged 250-control
standing acquisition and the same SONIC prefix. Source clips are not trimmed;
all 11 parent request records, including the unavailable original elbow, remain.

New `g1_true23_recovery_window.py` checks every pre-return physics state,
effort, time and warning record, plus active requested/applied targets and
previous-target history. Terminal state and previous target must match the
original full trajectory exactly; the first return step may not reset them.
No synthetic MjData reconstruction or target reseeding is used. This produces
**140 exact prefixes / 37,260 active prefix physics steps**, plus three exact
full-request control replays. The summary rejects omitted, reordered or
duplicated boundaries and does not assume monotone recoverability.

All probes retain the existing 500/50 Hz cadence, active/return gains,
5 rad/s slew, 0.05 rad hard-joint margin and 23.75%-of-table effort cap.
The compatibility standing actor retains its existing rounded gains; the
active native controller retains its full-precision gains. The modeled
35 Nm ankle table remains unverified as a manufacturer rating. Return still
requests all **250 controls / 2,500 physics steps**. This compatibility actor
is not a simulated or real native Unitree FSM handoff.

| Candidate | Tested dance boundaries | Successful five-second returns | First failed return: controls / physics steps | Original full dance / return |
|---|---:|---:|---:|---:|
| LoRA 100 | 1–60 | 1–14 (0.02–0.28 s) | after 15: 86 / 868 | 60/535; 0/250 |
| LoRA 300 | 1–56 | 1–13 (0.02–0.26 s) | after 14: 108 / 1,086 | 56/535; 0/250 |
| Original v14 100 | 1–24 | 1–2 (0.02–0.04 s) | after 3: 107 / 1,071 | 24/535; 0/250 |

There are **29 successful and 111 failed prefix returns**. Every failed
return has a nonempty immediate target interval at handoff. Of those failures,
**110** later encounter an empty effort/position/slew intersection; LoRA 300
after control 25 instead loses standing posture after 539 return physics
steps (minimum height 0.44736 m, maximum tilt 0.92980 rad). No candidate has
a successful later boundary after its first failed one in this observed set.

LoRA 100 after control 14 returns for five seconds with maximum drift
0.01891 m. After control 15, the initial tilt is only 0.10436 rad and the
smallest immediate interval is 0.02 rad wide, but return fails after 1.736 s.
Left ankle roll then has a 0.00097588 rad interval gap; horizontal drift has
reached 0.17647 m. This is an actual integration stop, not merely failure of
the provisional drift screen. Later handoff states can look more upright
yet recover worse: LoRA 300 after control 56 has tilt 0.00701 rad, but its
return integrates only 13 physics steps. Tilt alone, or a valid next target,
therefore does not establish recoverability for this controller.

### Verification and limits of this result

**641 tests pass** in 190.22 s across 49 focused modules, including 27 new
planning, state-history, prefix-completion and return-duration checks. The
same two legacy test-module asset-root substitutions remain; this is not a
whole-repository test claim. Ruff format and critical-error checks pass.

Independent `integrity_report.json` verifies **1,373 files with zero
mismatches**, preserving all 1,066 prior pinned files. It checks **143
continuous traces / 491,405 actual physics steps**, including **95,231 return
physics steps**, no engine warnings, exact adjacent state continuity and
recorded generalized force equal to applied effort. Active PD equations
agree within 1e-12 Nm; every phase stays within the existing effort cap.
Return targets are **reconstructed from recorded effort/state and known PD
gains**, not separately recorded by the legacy trace. Their reconstructed
slew and hard-joint margins pass the existing bounds, including the first
return step's continuity from the actual last active target.

Evidence root:
`artifacts/g1_true23_frozen_lora/recovery_windows_20260906_v1/`.
Candidate report SHA256 values:

- LoRA 100: `7e0d2269e57eb1ad17b08928265149d18ea641b80831e84a2c1ad0833ed5caa9`.
- LoRA 300: `641874c418bcd18e018264d316b85350d71c5b5caa8d44f423b40a21a619d51f`.
- V14 100: `cb6f3d9e366e6753be4b514b0efe390df7139fbe9fb239c6da091fbb5f0c4d0e`.
- Integrity: `2a4c0cbe9b6baf739709a8519eaddb6b5c05b4b2b857d6d583a37f1b13f1bb3a`.

Every prefix remains explicitly failed for full-source motion fidelity and
lifecycle qualification. No automatic live recovery guard, normal-mode
handoff, dance success or physical safety claim is promoted. Full-motion
tracking and stable recovery need improvement before another hardware test;
these early failed-return states now provide exact reproducible witnesses.
Training/replay contact-model differences remain open, as does the native
normal-mode handoff. Earlier one-step predictive and half-tempo failures
remain failed evidence, not untried fixes. Six absent axes still require
per-motion retargeting and qualification; exact reproduction of every
original 29-DoF pose is impossible on 23 DoF. Goal remains active.

## 2026-09-06 continuation: original-v14 corrected-controller comparison

**Still NOT ready for physical dance or live full-body teleop.** A separate
original-v14 method run completes **100 new PPO updates / 51,200 simulator
transitions**, using the same complete corpus, seed, native23 controller and
IEEE precision as the current LoRA comparison. Its full dance fails at
**16/535** reference-start controls and **24/535** historical-start controls,
with **0/250 return**. It also loses standing. The standing-retained LoRA
update-100 baseline reaches **64/535 / 60/535** and preserves simulated
standing; it is better on these measured cases but remains unqualified.
Reverting to the original trainer is therefore not a supported fix.

No robot connection, DDS, SSH, arming/mode command, physical motion or
deployment change occurred. The existing dirty hardware work remains
untouched. The physical bit-30 motors-off cause remains unknown; a simulator
constraint rejection does not explain that hardware fault.

### What is matched, and what is deliberately not relabelled

New `train_g1_true23_v14_native_ieee.py` wraps the existing v14 trainer without
editing its source. It keeps the original hash-bound recovery actor, fixed
0.10 exploration standard deviation, last hidden block/head training scope,
clip-contained uniform sampling (including v14's adaptive-sampling fallback),
original rewards and adaptive PPO learning-rate schedule. The only changes
from the original v14 implementation are the explicitly recorded
`native_support_stateful_v2` actuation profile and IEEE float32 boundaries.
Fresh-only execution rejects resume requests and checks frozen actor tensors
before every checkpoint. Both original and wrapper source files are bound in
the new **30-file training source manifest**.

V14 trains **274,455 parameters in four actor tensors**, versus the current
rank-8 LoRA's **253,944 parameters in 18 tensors**. Both use 32 environments,
16 rollout controls, five epochs, eight minibatches, initial learning rate
5e-6, seed 20260906, the same critic architecture and complete 5,940-frame
corpus. The standing clip remains separately labelled; unavailable original
elbow is not replaced. All native gains, 5 rad/s slew, 0.05 rad joint margin,
95%-of-quarter-effort projection and 500/50 Hz cadence remain unchanged.
The modeled 35 Nm ankle table is still not a verified manufacturer rating.

This is **equal new-update/transition budget, not a one-variable ablation**.
Actor initialization/pretraining, trainable layers, exploration, motion
sampling, learning-rate adaptation and standing retention differ. V14 has no
standing LoRA bootstrap or retention loss. It starts from recovery checkpoint
SHA256 `d13f47eff7348a7fce1277233a1d1795a2bafe12cd8000b2e101351c73c63bcc`;
its existing training history is not counted as new motion updates.
The separate four-environment smoke completes two updates / 64 transitions;
the full run does not resume from that smoke. The 100-update process exits
zero after 280.48 s. Only 100 of its declared 1,000 updates ran; 900 remain
unexecuted, with no training process left running. Final rolling reward
-96.55 and episode length 5.24 controls are not motion-completion metrics.

### Correctly paired full-request evaluation

New `g1_true23_v14_diagnostic_pair.py` uses the existing exact-policy ONNX
exporter/verifier, including its unchanged minimum-update gate. It rejects
wrong method, controller, precision and reference contracts, and rejects an
embedded safe-target transform that the shared controller would apply twice.
Both networks come from V14's same checkpoint; no released LoRA encoder is
borrowed. Update zero is not exported or evaluated by bypassing its gate.

`evaluate_g1_true23_v14_motion_ppo.py` reuses the original complete request
plan. A regression test checks its simulator-call arguments are AST-identical
to the current LoRA evaluator. All 11 records remain: eight original source
requests (one unavailable), the historical-start dance lifecycle, and two
standing prerequisites. Full source durations and all 23 active joints remain.

| Case | V14 100 | LoRA 100 | LoRA 300 | Requested controls |
|---|---:|---:|---:|---:|
| Hand crawling | 16 | 64 | 60 | 595 |
| Happy dance / reference | 16 | 64 | 60 | 535 |
| Happy dance / historical | 24 | 60 | 56 | 535 |
| PICO upright | 17 | 37 | 37 | 1,013 |
| PICO standing | 7 | 37 | 35 | 1,013 |
| PICO crouch | 20 | 25 | 23 | 1,013 |
| PICO walk 001 | 16 | 22 | 30 | 684 |
| PICO walk 010 | 11 | 24 | 28 | 499 |
| Synthetic standing / reference | 30 | 500 | 500 | 500 |
| Synthetic standing / acquired | 36 | 500 | 500 | 500 |

Every executed V14 case ends in `TargetIntersectionError` and fails full
motion fidelity. Both 250-control standing acquisitions pass, but neither
V14 lifecycle completes any of its requested 250 return controls. Those
transitions use the explicitly labelled compatibility standing actor, not
a native Unitree FSM handoff. LoRA's two standing successes do not count
as dance or teleop qualification; its historical dance return also remains
0/250. LoRA 300 is context, not an equal-budget comparator to V14 100.

At V14's reference-dance failure, right ankle pitch has a 0.000238621 rad
gap between the slew and effort intervals; the instantaneous target would
need 5.11931 rad/s instead of the unchanged 5 rad/s. At historical-dance
failure, left ankle pitch needs 5.31042 rad/s. These are observations at
already rejected states, not permission to raise limits or proof that faster
slew would produce a stable dance. Standing return is requested too late to
find a feasible target at those states.

Evidence root: `artifacts/g1_true23_frozen_lora/v14_native_ieee_20260906_v1/`.
Checkpoint SHA256:
`32538cb83e676877b5581f72dce05f9208eb2f5ee9f177d16c6ea223aa28aeb5`.
Lineage SHA256:
`f5a597537db704dc57e52085b51779d4b1e7b3b191670f42802cd1b8186ba81f`.
Full evaluation report SHA256:
`9040a749d21808eeb8197bcd283818306cc0e5e4540f2e5484328f876460a057`.
ONNX three-probe encoder error is exactly zero; decoder maximum absolute
error is 2.264977e-6, with paired action error 2.503395e-6. **614 regression
tests pass**, including 29 new comparator/pairing checks. Neither candidate
is selected for deployment. This covers 48 focused modules in the pinned WSL
runtime, with the same two legacy test-module asset-root substitutions as
the preceding regression run; it is not a whole-repository test-pass claim.

Independent `integrity_report.json` verifies **1,066 files with zero
mismatches**, retaining all 984 files from the previous audit. It verifies
all ten continuous traces / **6,950 actual physics steps**, no engine
warnings, exact adjacent state continuity, and agreement between recorded
efforts and the native PD equation within 1e-12 Nm. All active efforts remain
inside the unchanged 23.75%-of-table projection cap.

Update zero exactly matches the original recovery network plus the 0.10 std
pin. Update 100 changes precisely four actor tensors; the other **25/29**
remain bit-identical, including the encoder and std. The critic changes;
fresh Adam starts empty and reaches step 4,000 on all 12 actor/critic tensors.
The original adaptive schedule ends at **1e-5**, whereas LoRA stays at 5e-6.
V14's frozen-during-this-run encoder is **not** the released frozen-LoRA
encoder: all ten teleop-encoder tensors differ, maximum element difference
0.00210693. This independently confirms why borrowing the LoRA encoder would
invalidate the comparison. Integrity report SHA256:
`ef8ecd40ceaa24d2af0f1888783fc9b4507ec5ff9b0ee2b5155730f034c96b22`.

Next bounded check: measure standing-return recoverability from actual saved
states **before** the first full-dance constraint rejection. Preserve the
complete failed dance result; a prefix return is only a recovery-window
diagnostic, never shortened-dance success. Earlier one-step prediction and
half-tempo experiments already failed and must not be sold as new fixes.
Full-motion feasibility/tracking, training/replay contact-model differences,
native normal-mode handoff and fresh supervised hardware evidence remain
open. Six absent axes still require per-motion retargeting and qualification;
exact reproduction of every original 29-DoF pose is impossible on 23 DoF.

## 2026-09-06 continuation: temporal mapping audit; 300-update PPO still fails dance

**NOT ready for physical dance or live full-body teleop.** The new recorded-
state audit finds no large training/evaluator input or projection mismatch on
2,675 recorded calls. Another **200 PPO updates / 102,400 transitions** finish
from the previous update-100 weights/Adam state, in a separate output folder.
Neither new checkpoint qualifies: full dance reaches **63/535 at update 200**
and **60/535 at update 300**; both historical-start returns fail immediately.
Standing still passes **250/500/250 acquisition/active/return** in simulation.
No robot connection, DDS, SSH, mode/arming command, motor operation, deployment
export or hardware-limit change occurred. Pre-existing hardware edits remain
untouched. The physical bit-30 motors-off cause is still unknown.

### Actual temporal mapping witness, not another reset-only comparison

New `audit_g1_true23_recorded_training_boundary.py` and
`g1_true23_recorded_training_boundary.py` execute the production training
observation functions, MJLab gravity calculation and circular history buffers
on the **actual saved CPU evaluator states**. All calls from both prior
update-100 evaluations are retained: **675 attempted full-motion-prefix calls
and 2,000 standing calls**, including terminal partial intervals. Every saved
successful active 2 ms projection (**26,646 total**) and all 16 terminal rejections are tested
with the actual training projection function, on CPU and IEEE GPU.

Reference q9/proof q10 comes from each bound source motion; current q/dq and
previous applied targets come from the physics/action traces. No saved policy
input is used to fabricate its reconstruction. For acquired cases, the nine
startup observations remaining in H10 use targets explicitly reconstructed
from recorded PD effort, state and reported gains. Their target reconstruction
is labelled; it is not claimed to be a separately recorded target channel.
The original one-time reference alignment is independently reproduced.

Maximum absolute differences across the two devices:

| Boundary | Maximum difference |
|---|---:|
| Encoder 267 input | 4.7684e-7 |
| Policy history 930 | 2.3842e-7 |
| Discrete FSQ tokens | **exactly zero** |
| Raw 23-D action, same ONNX pair | 1.9074e-6 |
| Applied projected target | 2.4662e-7 rad |
| Applied PD effort | 1.7001e-5 Nm |

All declared tolerances pass; no successful evaluator substep is rejected by
the training projection, and every terminal rejection agrees. Training's
latched zero effort until its next 50 Hz reset is explicitly **not** a physical
standing-return strategy. The evaluator stops before integrating the rejected
substep. These different terminal responses are retained, not bypassed.

This witness supplies motion indices and recorded body-local angular velocity.
It does **not** execute a complete MJLab environment, automatic command-manager
stepping, real sensor sampling, observation corruption, random reset behavior,
physics or the robot. It therefore does not remove the previously measured
training/replay model differences or prove unrecorded-tail/hardware parity.
Both current observation paths use the legacy 18 cm native wrist proxy;
agreement between them does not establish original29 hand-frame equivalence.

Initial audit `_v1` rejected the older saved model's missing actuator-range
metadata before producing any case result. `_v2` fills the same fields as the
current evaluator on an **in-memory copy**, then requires the exact existing
compiled model hash `1f616be8...`. No source model or actual limit is edited.
The failed script/log are preserved. Successful witness evidence:
`artifacts/g1_true23_frozen_lora/recorded_training_boundary_20260906_v2/audit/`;
report SHA256 `30843e75f1cd92b98f82426adae4988cf084dd412c7ec6d49befac94068acca0`.

### Actual continuation and the resume limitation

Evidence root:
`artifacts/g1_true23_frozen_lora/ieee_motion_ppo_resume300_20260906_v1/`.
The new `breadth/` directory loads the unchanged, hash-checked IEEE update-100
checkpoint. Corpus, 32 environments, 16 rollout steps, five epochs/eight
minibatches, seed, IEEE policy, frozen platform, decoder LoRA, standing-retention
weight 10, reset/noise distributions and all actuation constraints are unchanged.
The 5,940-frame corpus still contains seven complete available SONIC/PICO
requests plus separately labelled synthetic standing, not qualified teachers.
No held-out standing rows enter training. Total PPO exposure is **153,600
transitions / 300 updates**; only 300 of the declared 1,000 updates have run.
The additional training process exits zero after 764.50 seconds. Its final
rolling episode length is 6.28 controls; this is not full-motion success.

**Correction to resume claims:** legacy frozen-LoRA checkpoints save actor,
critic, Adam, learning rate and counters, **not global RNG, simulator state
or adaptive-sampler state**. The new process restarts those states. Standing
anchor batches are reproducible from seed plus actual Adam step; the anchor
cache is regenerated and checked. That does not restore the global PPO random
stream. The raw continuation driver's RNG-restored metadata is incorrect and
is preserved as raw evidence; `experiment_report.corrected.json` and the
independent integrity report explicitly supersede it. Earlier references to
exact resume must not be read as bit-exact uninterrupted training equivalence.

### Full requested replays of both saved checkpoints

Paired encoder/decoder exports are independently validated. Decoder three-probe
ONNX maximum absolute errors are 1.66893e-6 / 3.33786e-6 at updates 200 / 300;
the frozen encoder remains unchanged. All original requests, whole source
durations, all 23 controlled joints, gains, 5 rad/s slew and quarter-effort/95%
projection remain. The 35 Nm modeled ankle table is still **not** a verified
manufacturer rating. No upper-body substitution or predictive filter is used.

| Requested case | Requested controls | Update 100 | Update 200 | Update 300 |
|---|---:|---:|---:|---:|
| SONIC hand crawl | 595 | 64 | 63 | 60 |
| SONIC happy dance, reference | 535 | 64 | 63 | 60 |
| Happy dance, historical posture + standing | 535 | 60 | 61 | 56 |
| PICO upright | 1,013 | 37 | 35 | 37 |
| PICO standing | 1,013 | 37 | 33 | 35 |
| PICO crouch | 1,013 | 25 | 22 | 23 |
| PICO walk 001 | 684 | 22 | 29 | 30 |
| PICO walk 010 | 499 | 24 | 24 | 28 |
| Synthetic standing | 500 | 500 | 500 | 500 |
| Standing after acquisition | 500 | 500 | 500 | 500 |

Every executed full-motion case fails `TargetIntersectionError` and full-clip
fidelity. Each historical dance requests 250 return controls but completes
**zero**. Separately, both standing acquisitions/returns complete 250 controls
each. That uses the pinned compatibility standing actor in simulation, **not
a native Unitree FSM handoff**. Unavailable original elbow remains unexecuted,
not replaced or counted as success. Both checkpoints are rejected for
deployment; neither is selected from training reward or isolated improvements.

**585 regression tests pass**, including 13 new trace validation, temporal
history, startup-target reconstruction and rejection tests. Tests retain the
same two previously documented original-asset-root substitutions.

Independent integrity verification checks **984 files with zero mismatches**,
including all 663 previously pinned files. The 20 new full-request/standing
traces retain **41,643 actual physics steps** with continuous pre/post state,
consistent engine time, zero engine warnings, and recorded PD-effort agreement.
Checkpoint 200/300 Adam counters are **8,000/12,000** for all 26 optimized
tensors; environment control counters are **3,200/4,800**. Source/config/lineage
and the IEEE standing anchor cache match the update-100 run exactly. Their
actor and critic tensor hashes change, confirming actual optimization.

Key SHA256 identities under the new evidence root:

- Update-200 checkpoint: `2953b7bcf610d51407377346d6435b9f80cf772707889cd45c6632926b216b15`.
- Update-300 checkpoint: `660cde84bcb8122baebe8f334872055e691f8374889d2d343710060a62b95c4c`.
- Update-200 decoder: `36787f0e963c2aa34a87224f34d1f0be53c80fc875c0734c7e907359f464af2b`.
- Update-300 decoder: `bdddab77692f36016a82070f59d639afcf95d13c100e770f50a9a9b851790117`.
- Integrity report: `f5acb2b7994e95a034e7ac89c4a2031b3369c6d516e84692c828842ffbce73f5`.

More identical training is not established as a fix. The corrected-budget
original-v14 comparison remains outstanding; the exact recovery checkpoint
required by that runner is locally present. Next comparisons must separate
the policy/conditioning problem from actual temporal-manager and mechanical-
model differences. Hardware testing still needs a qualified controller and
standing handoff; historical confirmations are not renewed authorization.

## 2026-09-06 continuation: explicit IEEE training tested; full-motion failure remains

**NOT ready for physical dance or live full-body teleop.** A separate IEEE
float32 run completes 100 real PPO updates from the same standing adapter.
Standing still completes **500/500 controls**, including simulated
**250/500/250 acquisition/active/return**. Full dance still fails at
**64/535**; historical-start dance reaches **60/535** but return remains
**0/250**. Correcting training precision alone does not solve full-motion
control or establish the cause of physical damping. No robot, DDS, SSH,
arming, mode, motor or hardware-limit operation occurred; dirty hardware
edits stay separate. Neither candidate is promoted or deployed.

### Precision is explicit, guarded and part of a new lineage

New `train_g1_true23_ieee_standing_retention.py` and
`g1_true23_training_precision.py` wrap the preceding standing-retention
trainer without editing its hash-pinned sources. They explicitly set IEEE
precision for global FP32, CUDA matrix multiplication, cuDNN and cuDNN
convolution/RNN operators. Other choices remain unchanged: cuDNN benchmark
on, deterministic algorithms off, original seeds, float32 actor/critic,
no CUDA autocast, same retention weight 10 and batch size 128. Conflicting
TF32 environment overrides reject. Construction, actor/critic forwards,
PPO updates, checkpoint load and save check the actual backend state.

An immutable `training_precision.json` receipt records the actual settings;
the same descriptor enters resolved configuration and exact resume lineage.
The pinned MJLab helper is identified by path and SHA-256. The source
manifest grows from 36 to **38 files**. This does not relabel or resume a
TF32 checkpoint as IEEE; the IEEE run starts a fresh PPO state from the
same checked standing-only adapter, with an independently resumed smoke run.
The backend configuration helper and precision settings are restored when
the process-local precision context exits.

The first launch correctly stopped **before any run directory or PPO
updates**: PyTorch's actual global default was `none`, while cuDNN operator
defaults still read `tf32` after MJLab set the parent to `ieee`. The initial
tests had normalized those defaults and missed that case. The helper now
sets each precision level explicitly; tests use the actual backend defaults.
That failed launch, its log and original helper snapshot remain under
`ieee_motion_ppo_20260906_v1/`. The successful experiment uses a new
`ieee_motion_ppo_20260906_v2/` directory. No failed updates are counted.
The explicit per-operator API follows the
[PyTorch 2.9 CUDA precision documentation](https://docs.pytorch.org/docs/2.9/notes/cuda.html#tensorfloat-32-tf32-on-ampere-and-later-devices).

### Same complete requests and training budget

The previous successful command arguments are reused except for the
additive launcher and new output paths. Smoke completes two updates and
genuinely resumes checkpoint 2 to update 4: **128 training transitions**.
The separate breadth run completes **100 updates / 51,200 transitions**
with 32 environments, 16 rollout steps, five epochs, eight minibatches,
`5e-6` learning rate and seed 20260906. Its configured budget remains
1,000 updates; no claim is made that those 1,000 updates finished.
The measured learning loop takes about 362 seconds; the entire breadth
subprocess, including initialization, takes 475.58 seconds. Final mean
completed episode length is 6.22 controls, still very short.

The same 5,940-frame corpus retains all seven available complete SONIC/PICO
clips plus separately labelled synthetic standing. Incomplete elbow remains
unavailable with no replacement. No motion shortening, tempo change, new
retarget, upper-body substitution, reset/reward/noise change, altered gains,
relaxed effort/slew/joint limits or extra physics settling was introduced.
The fixed-zero absent-joint codec, original FSQ/base weights and exploration
distribution remain frozen; only the same decoder LoRA parameters train.
Held-out standing arrays remain excluded from retention.

Both update-100 candidates are evaluated through their correctly matched
ONNX pairs with the same frozen encoder and unchanged full-request evaluator:

| Complete request | Retention PPO 100, TF32 training | Retention PPO 100, IEEE training | Requested controls |
|---|---:|---:|---:|
| Hand crawl | 65 | 64 | 595 |
| Happy dance, reference start | 64 | 64 | 535 |
| Happy dance, historical acquisition | 57 | 60 | 535 |
| PICO upright | 35 | 37 | 1,013 |
| PICO standing | 33 | 37 | 1,013 |
| PICO crouch | 26 | 25 | 1,013 |
| PICO walk 001 | 22 | 22 | 684 |
| PICO walk 010 | 24 | 24 | 499 |
| Separate synthetic standing | 500 | 500 | 500 |
| Separate standing after acquisition | 500 | 500 | 500 |

All eight executed full-motion cases still fail the target-intersection and
motion-fidelity checks. The extra stationary successes are prerequisites,
not successful dance/PICO clips. Simulated return uses the existing pinned
29-to-23 Unitree compatibility standing actor, **not physical native FSM
handoff**. Physical bit-30 motors-off remains unresolved. Exported decoder
maximum absolute error is `1.847744e-6` on the existing three probes only;
encoder FSQ-token export parity is exact.

### Actual input recording, not reconstructed or shortened motion

New `record_g1_true23_motion_ppo_inputs.py` and `g1_true23_policy_input_trace.py`
add per-inference 267-D semantic input, 930-D history, 994-D decoder input
and 23-D raw output to the same complete-request evaluation. Original input
arguments, returned arrays and policy call count are preserved. Terminal
inference attempts remain recorded, and unavailable outputs are explicitly
NaN with a returned/not-returned mask, never fabricated actions. The wrapper
adds no policy or physics calls and does not edit the shared evaluator.

Both candidates have fresh recorded full-request runs, including the
historical dance lifecycle and separate standing tests. These traces enable
direct arithmetic comparison on **actually encountered** motion states,
without inferring policy inputs from saved qpos or substituting standing
states for dance. Backend comparison and physics-equivalence results are
recorded separately from motion qualification.

Recorded arithmetic comparison covers **2,675 inference attempts**: 334 and
341 attempted motion-prefix states from the TF32- and IEEE-trained models,
respectively, plus 1,000 stationary states per model. Motion coverage stops
at each actual failure; this is not coverage of the unexecuted remainder of
each clip. All recorded inferences returned before the controller guards
stopped the failing motion runs. Both CPU PyTorch and IEEE GPU reproduce
the captured ONNX encoder tokens **exactly** at batch sizes 1, 32, 64 and 128.
Maximum raw-action error across both IEEE backends/datasets is `4.529953e-6`;
maximum safe-target error is **`1.072884e-6` rad**. Proprioception agrees
exactly. These are measured input-stream results, not universal backend or
hardware equivalence.

With TF32 enabled, batch 32 changes encoder tokens in **90/1,334** and
**63/1,341** rows. Within the motion prefixes alone, those counts are
**16/334** and **20/341**. Batch sizes 64 and 128 instead change 97 and
69 rows; batch size 1 changes none. A single-frame encoder-only check
would therefore miss this observed training-sized batch mismatch. Maximum
raw-action error reaches 0.0939923 and safe-target error **0.0410333 rad**
on the TF32-trained dance prefix. Feeding captured ONNX tokens directly
into the TF32 decoder reduces its worst raw error to 0.00189686 across
both datasets, separating decoder arithmetic from the encoder-token jump.
The IEEE GPU independently reconstructs the actual training standing anchor
cache exactly: `a93e6b92c957979b813f0188c4ffc629fcd27dc8daa87be8762a23f3c7d0d968`.

The measured arithmetic mismatch is removed on the tested states, yet the
same full-motion failures persist. Next investigate training-versus-evaluator
observation construction and controller/learning behavior using these saved
inputs and physics traces. More precision variants are not the next fix.
These checks do not by themselves prove that training observations, reset
states, rewards, actuation, timing or robot transport match deployment.

The expanded 45-module suite passes **572 tests**, no failures or skips, in
181.85 seconds: 14 precision/launcher checks and seven recording checks are
added to the previous 551. Only the same two legacy test asset-root
substitutions are used. Ruff lint and formatting checks pass.

Independent integrity audit verifies **663 bound files**, preserving all
539 previous bound inputs with zero mismatches. It checks 30 continuous
native23 simulations / **62,494 actual physics steps**, finite continuous
state, actual engine clocks, zero engine warnings, compiled-model identity
and unchanged outer effort bounds. Recorder runs preserve **720 original
simulator arrays exactly**, and their original result fields are identical
to their respective non-recording baselines. IEEE initial actor/critic
hashes exactly match the previous TF32 run. Smoke resume reaches 16 Adam
steps and 32 environment controls; breadth reaches **4,000 Adam steps for
all 26 optimized tensors** and 1,600 environment controls. Both IEEE runs
share the independently reproduced anchor cache. Actual saved breadth
configuration differs from TF32 only by the explicit precision descriptor.
No training or evaluation process remains running at this checkpoint.

Artifacts: `artifacts/g1_true23_frozen_lora/ieee_motion_ppo_20260906_v2/`.

- Actual IEEE checkpoint 100 SHA-256: `f20f82385dd7c652a7a74b6103a4753f0317b6fb10055452862d647e7dc14de5`.
- Adapter tensors: `1caefbc84496fc278089ff1fa7e3005e350d60611a3f1d52b002b19f9c0e52d1`.
- Merged policy tensors: `dd6e7ceaa29a42f86462a46ff68ecaaaafe88f2328193c3a0ef0f91ce2abfe3c`.
- Matched decoder ONNX: `c03d051221645dac1343c02287cc5d2ee7a76c74a192ea7da7edf46d6331dce0`.
- Full-request evaluation: `9479ab7c4ec0d81d5fe1efc52f579eac9fc79998822faa3f0f07a0616a7462dd`.
- Regression JUnit: `d3762c65dc21d32c46350624783265555c08d13685a8e4a821f00cc3251919e3`.
- Recorded-stream arithmetic comparison: `2f29be5730306237c5dd87b6658ab730abbac1c1bdd79e3f24f647e0c37b147f`.
- Independent integrity report: `2b983c9b42dc20f2e441748dc98517e131f2cd15f450af0fc3fa2f302a03314a`.

No new matched-budget original-v14 trial, physical dance, native standing
handoff or live-stream qualification has been established. Six absent axes
still prevent universal exact 29-axis reproduction. Complete-motion
task-space/contact/force qualification remains required on native23.

## 2026-09-06 continuation: standing retained through motion PPO; TF32 token mismatch isolated

**NOT ready for physical dance or live full-body teleop.** A new, separate
standing-retention PPO run preserves the previously passing stationary
behavior through 100 real motion-training updates. Both stationary tests
complete **500/500 controls**, including **250/500/250 acquisition/active/return**
for the acquired case. Every complete dance/PICO/crawl request still fails.
The resulting policy is diagnostic only: not selected, promoted or deployed.
No robot, DDS, SSH, arming, mode, motor or hardware-limit operation occurred.
The existing dirty hardware edits remain separate. Physical bit-30 motors-off
and native Unitree mode handoff remain unresolved.

### Joint standing-output retention, not substituted motion or a second optimizer

The new additive `train_g1_true23_standing_retention.py` wraps the checked
standing-only initialization. It retains the complete 5,940-frame motion
corpus and original sampler, while adding a decoder-output retention loss
on the three validated standing **training** episodes, 1,500 input states.
The separate held-out episode is hash-bound but its arrays are not loaded
by this loss. Frozen encoder tokens and proprioceptive inputs are cached;
anchor outputs come from the checked standing adapter itself. Neither
failed motion actions nor standing teacher actions become motion labels.

Each ordinary PPO minibatch receives the following extra term **before**
the existing actor/critic gradient clipping and the same single Adam step:

```text
10 * (mean(((safe_target(raw) - anchor_target) / action_scale)^2)
      + 0.01 * mean((raw - anchor_raw)^2))
```

Weight 10 and batch size 128 were fixed before evaluation. Anchor sampling
uses a separate CPU generator seeded from the actual Adam minibatch count;
it does not consume PPO's global random stream. Resume checks the exact
runtime/cache receipt, resolved lineage, optimizer and counters. Source
hashes include all three added implementation files. The used RSL-RL 5.0.1
PPO source is pinned; unsupported recurrent, adaptive, distributed, RND and
symmetry branches reject instead of silently changing their behavior.

Zero-weight tests reproduce the installed stock PPO's actor, critic,
optimizer, original losses and random state exactly on the tested CPU MLP
fixture, across value-clipping and minibatch-normalization combinations.
This is not a claim of bitwise full-model CUDA-training identity. The
frozen SONIC encoder, FSQ, base weights, exploration distribution and 23/29
codec remain unchanged; only the same 253,944 decoder LoRA parameters train.
No reset, reward, gain, action-noise, effort, slew, joint-limit or motion
amplitude change was made. Existing polish and teacher-admission gates stay.

### Actual training and full-request comparison

The smoke run completes two updates, then genuinely resumes checkpoint 2
to update 4: **128 transitions** total, four environments, eight rollout
steps, two epochs and two minibatches. The first 32-environment breadth
attempt fails with host `OSError: [Errno 12] Cannot allocate memory` while
loading the source core, before any PPO updates or checkpoint creation.
It overlapped the regression suite. Its failure/log are retained; a serial
retry in a new directory succeeds. No failed attempt is counted as training.

The actual serial breadth run completes **100 PPO updates / 51,200
transitions**, with the previous 32 environments, 16 rollout steps,
five epochs, eight minibatches, `5e-6` learning rate and seed 20260906.
The budget remains 1,000 updates, not completed. No training process remains
running. Final mean completed episode length is only 6.42 controls; neither
training loss nor reward establishes motion quality.

Both following update-100 policies use the same matched frozen ONNX encoder
and the unchanged full-request evaluator, sources and native23 simulator:

| Complete request | Previous PPO 100, no retention | Retention PPO 100 | Requested controls |
|---|---:|---:|---:|
| Hand crawl | 54 | 65 | 595 |
| Happy dance, reference start | 57 | 64 | 535 |
| Happy dance, historical acquisition | 46 | 57 | 535 |
| PICO upright | 35 | 35 | 1,013 |
| PICO standing | 34 | 33 | 1,013 |
| PICO crouch | 26 | 26 | 1,013 |
| PICO walk 001 | 29 | 22 | 684 |
| PICO walk 010 | 29 | 24 | 499 |
| Separate synthetic standing | 89 | **500** | 500 |
| Separate standing after acquisition | 70 | **500** | 500 |

The unavailable elbow request remains an explicit eleventh record with no
replacement. All eight executed full-motion cases fail with
`TargetIntersectionError` and failed fidelity. Historical dance return
remains **0/250**, so the stationary return result does not qualify return
after dance. Several full-motion results remain below the initial standing
adapter's earlier CPU results; this is standing retention, not general
motion improvement or SONIC parity.

Acquired stationary return completes **250/250**, with maximum tilt
0.0303163 rad and horizontal drift 0.0154050 m. This is the hash-pinned
Unitree 29-to-23 zero-velocity compatibility actor **inside the simulator**,
not the physical Unitree FSM and not a demonstrated native mode transfer.
Encoder export FSQ-token parity is exact and decoder maximum absolute
error is `2.384186e-6` on the existing three export probes only.

On the same 1,500 standing **training** inputs, CPU safe-target RMS drift
from initialization drops from **0.0405435 rad** without retention to
**0.00758414 rad** with retention; maximum drift drops from 0.106259 to
0.0284377 rad. Held-out arrays were not used for this measurement or tuning.

### Newly isolated backend mismatch: inherited TF32 changes discrete tokens

The inherited causal trainer calls MJLab's `configure_torch_backends()`
with its default `allow_tf32=True`. The previous standing-only fit disabled
TF32. An initial standalone default-FP32 probe therefore failed to reproduce
the actual training anchor cache. That failure is preserved, not suppressed.
A second probe uses the exact source helper and actual training settings;
its TF32 cache matches both training runs and the genuine smoke resume.

On identical 1,500 standing inputs and frozen source weights, measured
against CPU inference:

| GPU math | Differing encoder-token rows | Maximum raw-action error | Maximum safe-target error |
|---|---:|---:|---:|
| IEEE float32 | 0 / 1,500 | `1.4305115e-6` | `4.4703484e-7` rad |
| Actual training TF32 | 204 / 1,500 | 0.0549138 | 0.0301018 rad |

TF32 changes one token element in each affected row; proprioception agrees
exactly. This establishes a training-versus-CPU numerical mismatch on these
inputs, **not** its prevalence on complete motion streams, not the sole cause
of motion failure, and not the cause of physical damping. The next bounded
experiment must explicitly pin IEEE training precision in a new immutable
lineage and compare complete motions under otherwise unchanged conditions.
Do not silently change this run's precision or blindly resume its remaining
900 TF32 updates. Full 267/930-input GPU/CPU/ONNX stream parity still needs
measurement; the existing motion physics archives do not save those inputs.

### Local evidence and remaining readiness work

Artifacts: `artifacts/g1_true23_frozen_lora/standing_retention_ppo_20260906_v1/`.
The six added source/test files contribute **40 new tests**; the expanded
42-module regression suite completes **551 passed**, no failures or skips,
in 194.96 seconds. Two legacy tests use the same explicit original-asset
root substitution as before. Ruff lint and formatting checks pass.

Independent integrity audit passes **539 bound files**, preserving all 457
previous bound inputs, with zero mismatches. It checks 10 continuous native23
simulator traces / **20,798 actual physics steps**: actual engine clocks,
finite continuous state, no engine warnings, requested-versus-generalized
effort, unchanged outer effort bounds and compiled-model identity. Both
training manifests bind 36 source files. Initial actor/critic hashes exactly
match the previous run; smoke resume reaches 16 Adam steps and 32 environment
controls, while breadth reaches **4,000 Adam steps for all 26 optimized
tensors** and 1,600 environment controls. Saved retention events cover every
actual update. Breadth resolved configuration differs from the previous
experiment only by the explicit standing-output-retention descriptor.

Key SHA-256 values:

- Actual checkpoint 100: `2bfd05e514b5c77606b59ee51ee302e9a280d7aeacef0fbb5bf758b011835c28`.
- Adapter tensors: `f0444891ae6306b1eacf010bd95ce04c38785d6c7c057bf5d496bee7dac4d34f`.
- Matched decoder ONNX: `6e5fc97a1b3c2de114369aeb84afa93bf2c0c8e5ec764aa0bb74387ea7da4a6d`.
- Unchanged matched encoder ONNX: `3806b2b63ebadf4d6cbf9f79b7072f2bf27ab8eb8bc6a9b3042f97739cc5428a`.
- Full-request evaluation: `1894d2c5c862fcb7a2f6a783a56f1d408226c766aaca06373ab094a9a4ee44b6`.
- Matched-backend drift evidence: `a3d7c423dfbb57b49016a3cf9394c5a30ec9b10478ad167a2c2c2601647b8f1b`.
- Actual TF32 training anchor cache: `6429d8cd2d1c8870c426269ca4c11c463aee1f3ca2246b6f142fb76e2628ff87`.
- Regression JUnit: `c9f8fcd2b537e92f849b288a6ca0edcc743c11b5831230e63ae5773f0f3fc7d4`.
- Independent integrity report: `ca075e1b358f4813c9fc3358a3ed1ffed7f6ea9d19ea113590fbf76dd60b3410`.

No new matched-budget original-v14 training comparison exists. A 23-axis
robot cannot exactly reproduce independent motion on six absent axes;
full-motion task-space, contacts, forces, transition and live-stream
qualification remain required. Standing surviving motion PPO is a useful
prerequisite, not authorization to run this checkpoint on the robot.

## 2026-09-06 continuation: real motion PPO from standing; standing regression rejects update 100

**NOT ready for physical dance or live full-body teleop.** The standing-only
adapter now has a checked, separate PPO initialization path, and a real
100-update motion-training session has finished. The resulting checkpoint
**loses the previously passing stationary behavior** and still fails every
complete motion request. It is not selected, promoted or deployed. No robot,
DDS, SSH, arming, mode, motor or hardware-limit operation was performed. The
existing dirty hardware edits are preserved outside this commit. Physical
bit-30 motors-off and native Unitree mode handoff remain unresolved; simulated
target-intersection failures do not establish the physical damping cause.

### Adapter-only initialization is not a fabricated PPO resume

New `train_g1_true23_standing_warm_start.py` and
`g1_true23_standing_initialization.py` validate the previous standing fit,
adapter tensor/file hashes, frozen-platform contract and both complete saved
stationary lifecycle traces before importing decoder LoRA. Every bound fit
input is rehashed. Actual engine clocks, state continuity, phase lengths and
requested-versus-generalized effort are checked. Source/contract mismatch,
dirty optimizer/counters, nonfinite weights, changed standing evidence or
claims of hardware/full-motion qualification reject the import.

The import preserves a fresh critic, empty optimizer and zero PPO counters;
failure rolls the adapter back. A separate `standing_initialization.json`
receipt records what was imported. The descriptor enters the exact resolved
training lineage, and the new launcher/helper enter its source manifest.
The old launcher and checkpoint formats are unchanged. The wrapper uses the
existing load transport only to dispatch this explicitly separate import;
later real `--resume` calls retain ordinary exact-PPO-state validation. It
does not relabel the standing artifact as a resume or bypass polish/behavior-
bank admission. Only breadth with `native_support_stateful_v2` is accepted,
and initial training requires a separate empty run directory.

The imported adapter has 500 **supervised standing** optimizer steps but
checkpoint zero has **zero PPO updates**. Its tensor hash is
`60482bc1c129fc0bc2190b9a058e36a9bd86502cfac3178deaa4dff1b9aaf86a`;
its merged true23 policy hash is
`d8f048a50e12c85532cf531d7ec508e7d71f70cb4bb8ae0be3f5ebce8a18756e`.
The frozen SONIC encoder, FSQ, all base weights and analytic 23/29 codec
remain unchanged; only the same 253,944 decoder LoRA parameters are trainable.

### Complete requests are retained as RL references, not accepted teachers

`prepare_g1_true23_motion_requests.py` binds all seven available full source
motions plus a separately labelled synthetic standing prerequisite: **eight
training clips, 5,940 frames, 50 Hz**. These are the previously corrected-hand
recorded-source hand crawl/dance fits and all five contact-restored PICO clips.
They are not new fits of the later C++-observation source replay. The incomplete
elbow request remains explicitly unavailable with no replacement. No source
is shortened, retimed, re-retargeted or rebased; each full serialized span is
checked against its original arrays after the declared float32 conversion.

References request behavior from RL; they are **not force-qualified action
labels, a full-motion teacher bank, or evidence that the motion is physically
feasible**. Existing clip-boundary handling prevents rollouts crossing joined
segments. Synthetic standing is not counted as a successful dance/PICO clip.

### Real smoke and breadth updates, unchanged constraints

The smoke run completes two updates / 64 transitions (4 environments,
8 rollout steps, 2 epochs, 2 minibatches). The separate breadth run completes
100 updates / **51,200 training transitions** (32 environments, 16 rollout
steps, 5 epochs, 8 minibatches), learning rate `5e-6`, seed 20260906. This is
the first bounded 100-update session of a configured 1,000-update budget,
not a completed 1,000-update run. No training process remains running and
the remaining 900 updates must not be blindly resumed on this evidence.
Independent checkpoint inspection confirms zero initial optimizer entries,
1,600 final environment control steps and exactly 4,000 Adam steps for each
of the 26 optimized actor/critic tensors. The smoke run has 16 environment
control steps and eight Adam steps per optimized tensor. Adapter and critic
hashes change through training; source manifests and imported initial hashes
match exactly. The matching source manifests contain 33 files.

No action-noise scaling, reset-distribution change, reward change, controller
gain change, relaxed effort/slew bound, hidden physics settling or upper-body
override was introduced. The original frozen exploration standard deviation
remains approximately 0.38455 on average. Initial priming takes zero physics
steps and rejects zero full batches in both runs. Synthetic reset states and
their repeated initial histories are still not physically acquired states.
Completed training episodes remain short (final rolling mean 6.88 controls);
training reward is not a motion-quality or readiness result. Earlier
noise-off/reset-lift experiments also failed, so noise alone is not a
supported explanation or established fix.

### Same-backend comparison proves loss of standing, not an export artefact

The first attempted zero-update diagnostic materialization correctly rejects
`requires a trained update_count`. That failure is retained in its log; no
counter, checkpoint name or validator was falsified or weakened. Update 100
exports through the unchanged trained-diagnostic encoder/decoder path.
Encoder token parity is exact; decoder export maximum absolute parity error
is `1.639128e-6` across the existing three deterministic export probes.
These probes alone do not qualify a deployed runtime.

To avoid comparing different inference backends, both actual checkpoint zero
and checkpoint 100 are independently replayed through the **same CPU PyTorch
adapter path**, preserving all requests and limits. Update 100 is additionally
tested through its newly matched ONNX pair. Each backend test retains eight
original requests, the historical-start dance lifecycle, and two separate
stationary regressions: 11 records, one unavailable elbow, 10 actual runs.

| Complete request | Standing initialization, CPU | PPO 100, CPU | PPO 100, ONNX | Requested controls |
|---|---:|---:|---:|---:|
| Hand crawl | 68 | 54 | 54 | 595 |
| Happy dance, reference start | 66 | 57 | 57 | 535 |
| Happy dance, historical acquisition | 58 | 46 | 46 | 535 |
| PICO upright | 50 | 35 | 35 | 1,013 |
| PICO standing | 44 | 34 | 34 | 1,013 |
| PICO crouch | 26 | 26 | 26 | 1,013 |
| PICO walk 001 | 22 | 29 | 29 | 684 |
| PICO walk 010 | 28 | 29 | 29 | 499 |
| Synthetic stationary prerequisite | 500 | 89 | 89 | 500 |
| Stationary after acquisition | 500 | 70 | 70 | 500 |

Every full-motion attempt fails `TargetIntersectionError` and its full-motion
fidelity screen. Standing initialization passes its two stationary screens
and **250/500/250** acquisition/active/return; PPO 100 fails both stationary
screens and its return at **0/250**. Historical dance return remains **0/250**
for both checkpoints. Slightly longer walk fragments do not compensate for
lost standing or establish choreography parity. Equal CPU/ONNX completion
counts are not a claim of bit-identical closed-loop state or C++ deployment
equivalence. This is direct evidence that this motion-PPO session degraded
the retained standing behavior, not a diagnosis of which loss/reset feature
caused that degradation.

### Verification and local evidence

**511 regression tests pass**, no failures or skips, in 199.65 seconds:
486 prior tests plus 25 initialization, CLI-boundary, full-corpus and request-
preservation tests. Only the same two legacy test modules redirect asset paths
to the original repository. New source/test files pass Ruff. The new tests
do not claim real robot validation or a newly completed matched-budget v14
training comparison.

Independent saved-evidence audit passes **457 unique bound files with zero
mismatches**, covering training source/assets/configuration, checkpoint
lineage, full corpus spans, both inference paths and regression results.
All **30 new simulated traces** have continuous actual engine clocks, zero
warning counters and continuous finite pre/post states. Requested effort
equals actual generalized actuator force and stays within the unchanged
outer envelope, including partial failure intervals. Checkpoint zero exactly
reproduces **360 saved arrays across all ten previous stationary/full-motion
attempts**. This checks preservation of prior evidence, not new success.

Evidence root: `artifacts/g1_true23_frozen_lora/standing_motion_ppo_20260906_v1`.
Experiment data, checkpoints and local drivers remain ignored local artifacts;
the committed changes contain the reusable source/tests and these notes.

| Artifact under that root | SHA256 |
|---|---|
| `corpus/corpus.npz` | `e285ca71295bc047ed076abfeda9348be9567475f4fca21c30e112ac62976c06` |
| `breadth/checkpoints/frozen_lora_model_0.pt` | `eab3a9e886c58d4cf36e902cf0ff0a2f15a910eca288910b22728204f0294a96` |
| `breadth/checkpoints/frozen_lora_model_100.pt` | `e7fd6a62a93a32175116ab8b5af3136b4ad0aaa1c36bc296dcfa19a9fb7cd3a0` |
| `torch_0/report.json` | `e4a390eb09c37a040425ebe18735160cc1958d55053f081ad7030e1e49794b0f` |
| `torch_100/report.json` | `cc9aef6570467a2dd7f540f95f9eea97a471a66e110f2cd4dccfe505f14b6709` |
| `model_100/evaluation/report.json` | `ec32b4fcd7f9a5a99c6a47ff2e596076dbfc7df9a3d0f1d68347c355c4bcf2de` |
| `regression.xml` | `77081daa22ce327d40a45a3ded8c26658e54b572673d2ca94ffa96bd760bc199` |
| `integrity_report.json` | `4c389ae05963dec7ec5a9a0451f6bec191600b1577b704b8de09f39f5805db83` |

Rejected update-100 adapter tensor SHA256:
`3512be7846cad548884b00159d9e61a3d3df19d2ae5f3a3dbea067f49646722e`.
Its diagnostic decoder SHA256 is
`e766c317dd9b6e410fbb25976a8db573fb100bfb3631e676701d668d28a3b88a`;
its matching encoder retains frozen tensor SHA256
`3625edb10aabd266196702aefd464ad07c93847f2d1722a977e18ef2a0143990`.

Next work: isolate early PPO action drift on the validated standing training
states and the actual sampled reset states; add and verify an explicitly
standing-only retention/rehearsal mechanism without treating failed motion
actions as teachers or training on the held-out standing episode. Preserve
the entire motion request set and all original limits. Retest both standing
lifecycles and every complete motion before selecting any candidate. Such
retention training is **not implemented or proven by this continuation**.
Original-v14 matched-budget comparison, contact/force-valid original-motion
teachers, complete elbow source, live-input timing and supervised native-FSM
handoff remain outstanding. A 23-DoF morphology also cannot reproduce every
independent 29-DoF joint trajectory exactly; task-space retargeting and honest
per-motion feasibility evidence remain required. Goal stays active.

## 2026-09-06 continuation: standing-only LoRA bootstrap passes; full motion still fails

**NOT ready for physical dance or live full-body teleop.** There is now a
passing **simulated stationary** SONIC acquisition/active/return sequence.
This does not qualify dance return, Unitree mode ownership, a gantry launch,
or firmware recovery. No robot/DDS/SSH connection, mode command, hardware
gain, limit, interlock or deployed policy changed. Existing dirty hardware
work is preserved and excluded from this continuation's commit. The physical
bit-30 motors-off cause remains unknown; the findings below do not diagnose it.

### Extra effort headroom is not the controller fix

The old paired native23 dance fails while its instantaneous position/effort
interval still exists: previous target plus the unchanged 5 rad/s slew
constraint makes the intersection empty. At the reference-start right-ankle
failure, instantaneous required target slew is 5.405736 rad/s. This is a
diagnostic number, not permission to raise the configured limit. By return
time the previous command/state is already infeasible; changing the requested
target to a standing actor cannot restore the missing intersection.

New additive `g1_true23_interior_target_filter.py` tests preferred inner effort
bands at 100%, 95%, 85% and 70% of the existing outer envelope. Inner-band
failure uses the minimum-effort point **inside** the unchanged outer interval;
an empty outer interval still rejects, with its original joint-level details.
Standing acquisition/return, all 23 SONIC-controlled joints, action amplitude,
model, source frames, tempo, 35 Nm simulated ankle cap, 5 rad/s target slew and
quarter-effort guard remain unchanged. No headroom variant becomes a default.

All 32 executed full-motion cases fail; four incomplete-elbow requests remain
explicitly unavailable. No source is shortened or substituted. Historical
dance completion improves 16 to 70/535 at 95% headroom, but every historical
standing return still fails at 0/250. PICO results are mixed; 70% headroom
crouch loses absolute height rather than solving motion tracking. The 100%
variant reproduces **every** previously saved hand/dance/historical array
exactly. Actual engine clocks, all warnings and every physics pre/post state
are recorded, including partial failure intervals and lifecycle phases.

### Isolating the standing prerequisite

A pure stationary reference is generated from actual native23 FK at a neutral
pose placed using the eight foot collision spheres. It is labelled synthetic
standing only, never used to replace a dance or PICO request. The existing
correctly paired baseline100 SONIC policy completes only 68/500 active controls
from that start and 93/500 after a five-second standing acquisition; return
then fails immediately. The pinned Unitree zero-velocity compatibility actor
completes all 500 controls under the same effort/slew limits.

Direct teacher-label copying is invalid: two requested joint-controls lie
outside SONIC's finite raw-action safe-target image; the largest correction
is 0.070943 rad. `g1_true23_standing_bootstrap.py` adds an explicit, recorded
nearest-representable request followed by the **original** float32 safe-target
transform. Default inversion rejects unrepresentable labels. It does not
loosen the action bound, joint limits, force cap or physics target projection.

The original diagnostic collector's compiled model differed only in its
joint-level force-limit alignment. The new collector sets exactly the same
model limits as the SONIC evaluator: model SHA256
`1f616be811e72988f4764f74b8c2eb7f57b5ee87873ee36204bd882ff9cf96d7`.
The unmodified teacher's entire 5,000-step state trace and 500-frame
encoder/history arrays remain bit-identical after alignment. Requested and
actual generalized actuator force are identical; no hidden force clipping
explains its standing success.

The representable teacher passes four complete 10-second trials: nominal,
plus/minus 0.01 rad synthetic leg-joint perturbations, and a separately held-out
half-sized perturbation. Every original/projected request, causal 267/930
input, raw23 label, actual effort, engine time and warning is retained. These
are standing-only labels, not a qualified dance behavior bank or a robust
hardware recovery envelope.

### Decoder-only training produces a real stationary improvement

`fit_g1_true23_standing_lora.py` reconstructs the exact baseline100 adapter
against its approved frozen SONIC source before fitting. Initial CPU policy
versus paired ONNX maximum raw-action error is `7.152557e-7`. It trains only
253,944 decoder LoRA parameters (rank/alpha 8), with the encoder, FSQ, analytic
23/29 codec and every released base tensor unchanged. Three entire episodes
supply 1,500 training rows; the separate 500-row episode is not sampled for
training or checkpoint selection. The fixed run uses 500 Adam updates,
batch 128, learning rate `1e-4`, seed 20260906, and disabled TF32. The loss fits
safe target requests plus a small raw-action term to retain a correcting
gradient outside saturation. Original force/slew admission remains intact.

Held-out requested-target RMSE drops from 0.433755 to 0.010540 rad. More
importantly, fresh closed-loop native23 tests pass:

- Synthetic stationary start: **500/500** active controls, upright and
  stationary-reference fidelity screens passed; no return was requested.
- Five-second acquisition: **250/250**; stationary SONIC: **500/500**;
  five-second compatibility-actor return: **250/250**. Full provisional
  simulator lifecycle screen passes. Return maximum tilt is 0.025435 rad and
  horizontal drift is 0.004916 m, with unchanged limits.

This is CPU PyTorch adapter inference, not an exported/qualified deployment
ONNX. The weights-only `standing_lora.pt` has a distinct diagnostic header;
it is neither a PPO resume nor a promotion-eligible hardware artifact.
Frozen-platform tensor hashes are checked before and after fitting and replay.

### Standing does not solve motion-conditioned control

`evaluate_g1_true23_standing_lora_motion.py` replays the complete previously
bound request set with the new adapter. It retains all five contact-restored
PICO clips and both complete recorded-source SONIC fits; missing elbow remains
missing. No standing reference, upper-body override or reduced-amplitude action
is substituted. These remain the earlier corrected-hand recorded-source fits,
not new fits of the later C++-observation source replay.

| Full request | Paired baseline100 | Standing-only adapter | Requested controls |
|---|---:|---:|---:|
| Hand crawl | 48 | 68 | 595 |
| Happy dance, reference start | 45 | 66 | 535 |
| Happy dance, historical acquisition | 16 | 58 | 535 |
| PICO upright | 48 | 50 | 1,013 |
| PICO standing | 38 | 44 | 1,013 |
| PICO crouch | 27 | 26 | 1,013 |
| PICO walk 001 | 17 | 22 | 684 |
| PICO walk 010 | 11 | 28 | 499 |

Every full-motion attempt still fails its effort/position/slew intersection
and full-motion fidelity screen. Dance now first fails at right hip pitch;
the historical dance return still completes **0/250**, unlike stationary
return. This is partial progress, not a dance fix or SONIC parity. Correct
matched-budget original-v14 training comparison remains undone.

### Verification, artifacts and next work

**486 regression tests pass**, no failures or skips, in 185.95 seconds.
This includes 43 new numeric, boundary, history/phase integration and
label/gradient checks (21 + 8 + 14). Only the same two legacy test modules
redirect asset paths to the original repository. All eight new Python
source/test files and five local experiment/audit/test drivers pass Ruff.
Independent saved-array checks rehash **250 unique bound files with zero
mismatches**. All **50** new recorded simulator traces have uninterrupted
engine time, zero warnings, continuous state and identical requested/actual
effort inside the unchanged outer envelope. Full original source requests and
the exact old baseline arrays are independently rechecked.

Local evidence root:
`artifacts/g1_true23_frozen_lora/interior_effort_20260906_v1`.
Headroom report SHA256:
`a9b91e753111a3541605bac9177043fac67e732bba03bd48d7b14ca2c140d8e1`.
Representable-standing report SHA256:
`7adc1c626ac577ee33ba1301b3b10163b716a3431bdf73e5ae4b86dc12ff683c`.
Standing-fit report SHA256:
`5411ee4b95d8003ffc50047f01e6535ac95138b37852d3a2a88ef6f89e874f52`.
Diagnostic adapter file SHA256:
`2c515b3609ee102f48757a7258f59601c6552c204f4e552b9936f0f6d0ee99e4`.
Adapter tensor-state SHA256:
`60482bc1c129fc0bc2190b9a058e36a9bd86502cfac3178deaa4dff1b9aaf86a`.
Full-motion adapter comparison report SHA256:
`a4acda3d8c7f1fc07f1216e28d9958bd8898e688975987e3d069b08fc2ec7a7a`.
Full integrity report SHA256:
`720edd7cf2890ca1a074fa6851aee7a233cf0eaef660eff8e5b910611130cd22`.
JUnit SHA256:
`1a16d4edbfad60f07f25e68a05d6c2869bc4a2a8f8b19a4ea549a97cb2bdd26d`.
Large reports, traces and weights stay local; implementation and tests are
committed. No existing artifact or deployed checkpoint is overwritten.

Next: integrate this standing-only prerequisite into a provenance-preserving
motion-conditioned training curriculum, retaining full original/PICO requests
and independent force/contact/fidelity checks. Do not train against the failed
crawl/dance teachers or treat stationary success as live readiness. Correct
29-to-23 retargeting cannot reproduce motions requiring absent joints exactly;
unsupported motions must remain explicit. Live-input qualification, actual
control ownership/normal-standing return and supervised physical evidence are
still required. Goal remains active; no physical dance is authorized by these
simulator results.

## 2026-09-06 continuation: captured C++ observations and actual engine-time replay

**NOT ready for physical dance, standing return or live full-body teleop.**
This continuation adds an offline numerical-boundary implementation and six
recorded source runs. No robot/DDS connection, mode call, deployed policy,
native23 controller, hardware gain, limit or interlock changed. No new native23
training or matched-budget original-v14 benchmark was run. The physical bit-30
motors-off cause remains unknown; the observation finding does not diagnose it.

### Missing history had different gravity semantics

The captured C++ `StateLogger::makeZeroEntry_()` pads unavailable history with
quaternion `(0,0,0,0)`. Its `quat_rotate_d` assumes a unit quaternion: feeding
that padding through the gravity gatherer yields `(0,0,+1)`, not Python's
`(0,0,0)` history padding. Thus the first policy call contains nine upward
gravity entries followed by the actual gravity measurement. This is legacy
padding behavior, **not a valid sensor rotation or a recommended hardware fix**.
The upstream [logger](https://github.com/NVlabs/GR00T-WholeBodyControl/blob/main/gear_sonic_deploy/src/g1/g1_deploy_onnx_ref/src/state_logger.cpp)
and [rotation helper](https://raw.githubusercontent.com/NVlabs/GR00T-WholeBodyControl/main/gear_sonic_deploy/src/g1/g1_deploy_onnx_ref/include/math_utils.hpp)
also show these semantics. The experiment binds the checked-out source copies,
not an assertion that every current upstream file is byte-identical.

New `g1_sonic_cpp_observations.py` captures ten complete, unchanged C++ methods,
the real logger, motion storage and math dependencies, plus the actual YAML
and observation registry. A standalone C++20 shared library executes these
methods on injected arrays. It checks the G1-mode 1,762-element encoder and
994-element decoder layouts, preserves the LowState float32 boundary and
double default-angle subtraction, and rejects changed layouts or invalid
state. CSV logging is disabled. No G1Deploy instance, SDK, DDS or command
writer is built or opened. Heading delta is zero; upper-body override is off.

Checks confirm oldest-first history, term ordering, joint permutation, future
frame clamping, 6D orientation layout and paused playback behavior. The paused
case repeats the current pose/orientation and zeros reference joint velocity.
An additive trace provider uses these actual gatherers without patching shared
modules. Existing Python simulation, history conventions and saved evidence
remain unchanged. Reference resampling, initial pose, nominal effort clipping
and the C++-parameter PD loop are retained explicitly. This does not execute
the live receding-horizon planner, CSV reader, runtime scheduling or firmware.

### Same recorded states isolate the input difference

Each old C++-parameter trace is replayed through the new observer using its
exact recorded states and previous actions. The newly inferred actions are
recorded but never executed by this comparison. Rejected final inference
calls remain present: each old elbow case compares all 134 calls, not only its
133 completed transitions. Source/reference/state/action/token inconsistencies
are rejected. Both old and new ONNX inputs and outputs remain available.

Across all six comparisons, future encoder-input differences are at most
`8.9407e-8`; all released encoder outputs are numerically identical. History
angular velocity, joint velocity and previous action are identical. Joint
position differs by at most `1.1921e-7`. Gravity differs by 1.0 in the initial
nine calls, then at most `1.1921e-7`. Maximum raw action differences during
startup are 1.227214 (hand), 1.235639 (elbow), and 1.239447 (dance); after nine
calls the worst difference across cases is `2.3842e-6`.
These are raw action units, not radians or torque. The paired comparison
isolates observation differences; it does not prove that startup padding alone
explains later physical behavior or that copying it would improve native23.

### Complete source attempts, not a successful shorter suite

`record_g1_sonic_cpp_observation_replay.py` runs every original clip under both
previously bound source models, with the same released encoder/decoder pair.
All 26 trace arrays retain the full attempted pre/post states, commands,
inferences and physics transitions. Four new arrays record actual engine time
before/after every substep and all eight MuJoCo warning counts/last-info values.
Clock discontinuity or any warning prevents a successful-completion label;
synthetic sampling labels alone cannot conceal an engine reset.

| Source model / clip | Old Python observations, completed/requested | Captured C++ observations, completed/requested | New maximum nominal-root error |
|---|---:|---:|---:|
| Original masks / hand crawl | 606/606 | 606/606 | 2.193242 m |
| Original masks / elbow crawl | 133/606 | 126/606 | Incomplete |
| Original masks / happy dance | 546/546 | 546/546 | 0.663665 m |
| Enabled hands / hand crawl | 606/606 | 606/606 | 1.879026 m |
| Enabled hands / elbow crawl | 133/606 | 126/606 | Incomplete |
| Enabled hands / happy dance | 546/546 | 546/546 | 0.687778 m |

Root errors are first-position-translation aligned at pre-command time, not
new native23 tracking results. Original-mask hand error was 2.248922 m;
enabled-hand error was 1.963423 m. Dance previously had 0.656584 m error under
both models: this boundary correction does **not** improve dance fidelity.
New elbow runs reject their 127th inference at Isaac index 27: raw actions
10.036927 and 10.125465 exceed the unchanged absolute bound of 10. Both failed
calls and all requested frames remain represented. No elbow teacher is emitted.

Every run has uninterrupted 2 ms engine time and zero warning counters.
Hand/dance final times are 12.12/10.92 seconds; elbow stops at 2.52 seconds.
This rules out a MuJoCo clock reset in these runs, not a physical firmware
fault. Source maximum physics joint speeds remain as high as 24.758 rad/s
for enabled-hand crawl and 13.632 rad/s for enabled-hand dance; these are not
native23 qualifications under its unchanged limits.

Original-mask crawl still puts visible hand vertices 146.960 mm below the
floor. Enabled-hand crawl reduces this to 10.943 mm, with hand-floor contacts
on 437 pre-control frames and nonfloor hand contacts on three frames. Unlike
the previous observation profile, enabled-hand elbow/dance each have one
pre-control nonfloor hand-contact frame. New dance traces therefore cannot
reuse an assumed identical-model result: they first differ at physics step
42 (zero-based). No new source trace is accepted as a teacher; no geometric
refit or full eight-clip SONIC/PICO qualification is claimed here.

### Clip end is not normal-standing handoff

The checked-out and [upstream controller](https://raw.githubusercontent.com/NVlabs/GR00T-WholeBodyControl/main/gear_sonic_deploy/src/g1/g1_deploy_onnx_ref/src/g1_deploy_onnx_ref.cpp)
distinguish nonplanner clip completion from process shutdown. Clip completion
clears play, resets frame 0 and requests heading reinitialization; it does not
switch to Unitree's native standing FSM. `Stop()` explicitly emits a damping
command. Neither path is proof of a qualified native23 normal-standing return.
The new replay tests observation pause behavior but does not claim to execute
that end-of-clip state machine or an actual ownership handoff.

### Verification and next work

**443 regression tests pass**, no skips, in 177.32 seconds; this includes 25
focused observer/trace/admission checks. Only the same two legacy asset-dependent
modules redirect asset paths to the original repository. All four new Python
source/test files and both local audit/test drivers pass Ruff. An independent
saved trace check rehashes **109 unique bound files with zero mismatches** and
rechecks engine time, state continuity, first-call padding and inference lineage.
Existing dirty hardware work remains untouched and outside the new commit.

Local evidence directory:
`artifacts/g1_true23_frozen_lora/original29_cpp_observation_replay_20260906_v1`.
Main report SHA256: `d6a3801b12c41b9ce3aca6e1209c0629213e0ad3a79788ab2ed81b31aeb9238d`.
Integrity report SHA256: `579ae88de78e98b91d0823c1807ee4ea4e7a010b3ea7454bb8a279704aaf4b67`.
JUnit SHA256: `9b1b3dbb9f8f49694bca384505bfb6083169ff667a1334161f0ec2e1095a50d7`.
Captured observation binary SHA256:
`84e4ed204ab677dc533eda1cb8eeb1047dbccf7991128902ecc69c01a9e95faf`.
These large local artifacts are not committed to Git; reproducible recorder,
observer, witness and regression source are committed.

Next priority is actual playback/start/stop and live-planner boundary parity,
then a physically feasible native23 controller/reference and training corpus.
The prior native23 dance still fails at 45/535 transitions and standing return
at 0/250; this source-only experiment does not replace those results. Correct
matched-budget v14 comparison, preserved five-PICO motion qualification, live
input testing and operator-supervised hardware/standing return remain undone.
No padding change, source completion or software test permits another physical
dance by itself. The goal remains active.

## 2026-09-06 continuation: neutral-wrist hand frames and collision-enabled source replay

**NOT ready for physical dance, standing return or live full-body teleop.**
All work below is offline. The new coordinate correction improves reference
fitting, not controller qualification. No robot connection, mode command,
deployed policy, hardware gain, limit or interlock changed. The physical bit-30
motors-off cause remains unknown; neither finding below diagnoses that event.

### The same 18 cm offset referred to different wrist frames

The existing source hand task uses `(0.18, +/-0.025, 0)` on wrist-yaw; its
native23 counterpart uses the same numbers on wrist-roll. The removed source
pitch/yaw links translate the hand frame by `0.038 + 0.046 = 0.084 m` at zero
wrist pitch/yaw. Independent FK confirms an **84 mm neutral task mismatch** on
both hands, although the shared wrist-roll frames coincide.

New additive `g1_true23_hand_frame_tasks.py` derives the source neutral proxy
in wrist-roll coordinates: `(0.264, +/-0.025, ~0)`. It validates the exact
29/23 layouts, removed-link ancestry, shared neutral frame and identity neutral
hand rotation. Unknown frame conventions are rejected, not silently calibrated.
Actual source trajectories retain all 29 joint angles, including missing wrist
motion. Only the two target points change in the diagnostic all23/SE(3) fitter;
its source targets, weights and physical/temporal bounds remain unchanged.

The meshes support the frame distinction but are not identical: source rubber
fingertips extend about 257.329 mm from wrist-roll, versus 253.315 mm on native23.
Source-to-nearest-native-vertex distances reach 8.861 mm, with approximately
2.4 mm median. Those are vertex comparisons, not continuous surface distances
or a proof of equivalent contact geometry. The 264 mm task proxy remains
outside the physical fingertip: **it is not a contact landmark**.

The explicit convention is `native23_source_neutral_wrist_hand_proxy_v1`.
Global `DEFAULT_TASKS`, VR offsets, native124/causal wire features, checkpoints,
training and live PICO interfaces remain unchanged. Serialized references
rebuild causal packets under the existing convention and disclose that fact.
A future policy-input change needs a separately versioned end-to-end migration;
these artifacts cannot silently redefine the inputs to an existing checkpoint.

Frame/mesh evidence: `artifacts/g1_true23_frozen_lora/original29_hand_frame_audit_20260906_v1/report.json`,
SHA256 `bc1be80e7ae5de523eb6aebaa565373ba2cdffd61d813d8cc98ca2c3986f88f0`.

### Source hand collisions tested without editing the original model

New `g1_sonic_hand_collision_variant.py` copies and recompiles an `MjSpec`,
following the [MuJoCo model-editing interface](https://mujoco.readthedocs.io/en/3.5.0/programming/modeledit.html).
It enables exactly the two previously visual-only rubber-hand meshes. Simply
patching compiled collision masks is not used: mesh collision graphs and BVH
data must also be rebuilt. All noncollision numeric model arrays, raw mesh
vertices/faces, masses/inertias, joints, gains/limits and solver settings are
checked unchanged. Native23 is not edited. The masks enable hand-floor **and
hand-self** collision; this is not a floor-only experiment.

Untouched source compiled hash: `40bd428f7a2e090a3bb2f85f61e965be179b6ab91b63d6f1a827733cf6a0cd96`.
Separate variant hash: `ff7f4e95bdf160abf98c83ea6098b68e38ee1903e0377ef7a66879ade084cbd8`.
Both models and every source dependency are retained and rehashed.

The released encoder/decoder, captured C++ parameters, source clips, timing and
unchanged original29 action/effort bounds are reused. Hand crawl completes
606/606 frames; worst visible-hand floor penetration drops from 145.947 mm to
12.149 mm, with active hand-floor contacts on 420 frames. No hand-self contacts
appear in the recorded pre-control poses. Elbow remains incomplete at 133/606
on the same action bound. Dance completes 546/546 with no hand-floor contacts;
its source execution is unchanged. Hand crawl still deviates from the nominal
planner root by 1.963423 m. Completion is not stock SONIC parity or a valid
native23 teacher. This still uses legacy observation/playback algorithms, not
verified full C++ deployment equivalence; actual engine time/warning counters
are not newly audited by this experiment.

Evidence: `original29_full_hand_collision_20260906_v1/report.json`, SHA256
`d704edbe820d4bc8ed132bf889f924be5ded145eeb6cbbca4ab670938ef6f48d`.

### Same-source fit comparison uses the same task metric

New `retarget_g1_true23_hand_frame_trace.py` retains all three requests under
either explicitly labelled source model. Incomplete elbow cannot be cropped,
padded or replaced. Both complete clips retain all 606/546 frames and all 23
joints at original tempo. FK, causal rebuilding and serialized 5 rad/s,
80 rad/s^2 bounds pass. The old-mask fits accept 24 updates but exhaust their
iteration budget; no convergence or contact-force qualification is claimed.

| Recorded original-mask source | Old fit, evaluated with corrected proxy | New fit, same corrected proxy | New worst foot error |
|---|---:|---:|---:|
| Hand crawl | 55.863 mm | 24.690 mm | 1.296 mm |
| Happy dance | 94.243 mm | 31.756 mm | 2.180 mm |

These compare each clip to the exact same recorded source bytes and models,
not nominal-planner fidelity or full-body closed-loop tracking. Under the old
numeric-offset metric, hand/dance errors instead change from 64.232/63.339 mm
to 92.227/101.164 mm. Both conventions are reported, so coordinate changes
cannot be passed off as an improvement under the old metric.

Contact still fails: old-mask hand fit overlaps the floor on 596 frames, up to
160.256 mm; dance overlaps on 477 frames, up to 10.998 mm. The separately
collision-enabled source hand fit stalls after 22 accepted updates (iteration
23 rejected), retains all 606 frames, but has 150.445 mm hand error and 79.316 mm
floor overlap on 392 frames. Solver stalling is not proof of physical
infeasibility. Its complete dance output is byte-identical to the old-mask
corrected dance, SHA256 `3fe0a95b3e33e291fffde0fc3ede0cfc2fbae3b5ca19e8a43fbbe72fe3d16251`.
None is accepted for training, and no full-eight SONIC/PICO manifest is written.

Evidence: `original29_neutral_hand_frame_fit_20260906_v1/report.json`, SHA256
`c28d730dedfa68af92ce5d63f0d60974a6ff7fa0fac8846f8fb46f279f2d3e4d`;
`original29_hand_collision_neutral_fit_20260906_v1/report.json`, SHA256
`19583e5aff527bcaf6fbdec4eb89443717a1a08cea45239c749125c73d193175`.

### Corrected references still fail the unchanged paired controller

The fixed baseline100 pair (`3806b2b6...` encoder / `90e27f37...` decoder)
executes four distinct cases. Old-mask and collision-source hand fits both fail
at 48/595 transitions. Dance fails at 45/535, historical-posture dance at
16/535. Every failure is an empty effort/position/slew target intersection;
all partial physics substeps remain recorded. The two byte-identical dance
cases from the second source explicitly reuse the first results, not two new
executions. Both incomplete elbows remain represented as unavailable.

Historical-start acquisition passes 250/250 transitions and 2,500 physics
steps. Return fails at 0/250 and zero physics steps: right ankle needs an
instantaneous 7.1197 rad/s target slew against the unchanged 5 rad/s bound.
This is a different failed trajectory from the prior 14.8714 rad/s return,
not a solved handoff. Historical posture is not fresh robot state, and the
standing compatibility actor is not a native Unitree FSM transition. PICO
clips were not rerun; this is not a complete eight-clip qualification.

Evidence: `original29_hand_frame_envelope_20260906_v1/suite_summary.json`, SHA256
`b2f7812462c3b7fc709af4edfb59d1601c827fbe3c3251df0138b45b6b32793c`.

**418 regression tests pass**, no skips, in 173.75 seconds, including 20 focused
collision/frame checks. The two legacy asset-dependent modules use the original
repository's assets. All six new Python source/test files pass Ruff; both new
local geometry/replay artifact drivers pass Ruff. The saved JUnit result is
`original29_hand_frame_audit_20260906_v1/regression.xml`, SHA256
`84c1a5e056da4d1129ad56c5baf5bed6209890f63415ea5f131af90d4409ad8a`.
All 114 unique bound inputs across these five and the previous four reports
were rehashed with zero mismatches. Existing dirty hardware work is preserved.

Next work remains source observation/playback equivalence, actual contact and
controller-feasible all23 motion/training, corrected matched-budget v14
comparison, safe standing handoff and operator-supervised live-input/hardware
qualification. Do not repeat a physical dance on the strength of geometric
fit improvements or passing software tests. The goal remains active.

## 2026-09-06 continuation: recorded original29 baseline and contact-model mismatch

**NOT ready for physical dance, standing return or live full-body teleop.**
All work below is offline. No DDS connection, robot command, hardware gain,
interlock, deployed controller, training run or policy promotion changed.
The physical bit-30 motors-off cause remains unknown. These findings concern
simulation/reference correctness, not a diagnosis of that physical event.

### C++ parameters were not reproduced exactly by the legacy simulator

The old Python original29 simulator uses half the C++ damping on hardware
indices 4, 5, 10, 11, 13 and 14: both ankle pitch/roll pairs and waist roll/pitch.
The checked-out and upstream [SONIC policy header](https://github.com/NVlabs/GR00T-WholeBodyControl/blob/main/gear_sonic_deploy/src/g1/g1_deploy_onnx_ref/include/policy_parameters.hpp)
use `2 * DAMPING_5020` there. Smaller differences also exist: C++ gains are
float32, defaults/scales are double, and `CreatePolicyCommand` casts its final
motor targets to float32. Legacy Python uses different precision boundaries.

The old simulator and its historical evidence are preserved unchanged.
New additive `g1_sonic_cpp_parameters.py` compiles a standalone arithmetic
witness against an exact saved header copy. It records the compiler, source,
binary, parameter JSON and hashes. Ten complete action vectors match the
compiled C++ target calculation exactly, including permutation and rounding.
No Unitree SDK or hardware executable is built by this witness.

New `g1_sonic_original29_trace.py` records either the exact legacy simulator
bytecode with private instrumentation or an explicitly labelled C++-parameter
variant. Both retain all physics substeps, pre/post states, requested and
actual generalized actuator forces, ONNX inputs/outputs and final state.
The C++-parameter variant still uses the legacy observation/playback algorithms
and nominal-effort clipping. It is **not full C++ deployment, firmware, mode
transition, or hardware equivalence**. No native23 limits were transferred to
the original29 run, and no native23 limits were increased.

`record_g1_sonic_original29_baseline.py` records all three original clips under
both profiles with the released encoder/decoder hashes, exact ABI, one-thread
CPU ONNX Runtime, and a saved compiled MuJoCo model. Pre-command samples are
compared at the same command time; post-control samples retain their explicit
20 ms offset. Raw planner timing and all requested frames remain recorded.

| Clip | Legacy completed / requested | C++-parameter completed / requested | C++-parameter maximum joint speed, all physics steps |
|---|---:|---:|---:|
| Hand crawl | 606 / 606 | 606 / 606 | 13.023 rad/s |
| Elbow crawl | 133 / 606 | 133 / 606 | 27.663 rad/s |
| Happy dance | 546 / 546 | 546 / 546 | 13.009 rad/s |

Elbow stops on the unchanged action bound, not successful full-clip completion.
The rejected extra ONNX call is retained. Six cases are recorded, but the
suite's `all_clips_completed` is false. Completion here remains survival only.
C++-parameter hand/dance nominal-planner root errors are 2.248922 / 0.656584 m
on the same-time pre-state comparison. Their active-geometry floor overlaps
reach 16.659 / 5.718 mm. None is an automatically valid native23 teacher.

Evidence: `artifacts/g1_true23_frozen_lora/original29_recorded_baseline_20260906_v1/report.json`,
SHA256 `ba3f74b9fe84325f21673dc095840ab1db5673c62f26e1c8a2ab6194d86b6932`.
The compiled source model is `40bd428f7a2e090a3bb2f85f61e965be179b6ab91b63d6f1a827733cf6a0cd96`.
Runtime: Python 3.11.15, NumPy 2.3.4, MuJoCo 3.5.0, ONNX Runtime 1.28.0.

### Complete recorded clips fitted with all 23 actual joints

New `retarget_g1_true23_original29_trace.py` validates complete contiguous
same-time sources and applies the existing whole-clip SE(3)/all23 fitter.
It rejects incomplete, shifted, cropped, retimed or falsely promoted sources.
Both full hand and dance clips accept 24 optimization updates, retain 606/546
frames and pass serialized FK, causal and unchanged 5 rad/s, 80 rad/s^2 checks.
The optimizer reaches its iteration budget; that is not convergence or teacher
qualification. Elbow is explicitly retained as failed, with no padded or
substituted output. PICO references are untouched; no accepted full-eight
training manifest is written.

| Complete clip | Worst foot error against recorded29 | Recorded29-relative root error | Nominal-planner-relative root error | Native23 floor-overlap frames / worst depth |
|---|---:|---:|---:|---:|
| Hand crawl | 1.206 mm | 0.036782 m | 2.236327 m | 597 / 191.225 mm |
| Happy dance | 2.265 mm | 0.053816 m | 0.645636 m | 527 / 10.726 mm |

Both foot screens pass against the recorded policy execution. Dance improves
from its own projected seed's 20.222 mm foot error to 2.265 mm. This is a
different source comparison from the preceding nominal-planner fit; do not
present the old 8.501 mm and new 2.265 mm as the same fidelity metric. Worst
hand-point errors remain 64.232 / 63.339 mm. Matching a virtual task point is
not evidence of physically valid hand contact.

Evidence: `original29_recorded_native23_fit_20260906_v1/report.json`, SHA256
`fc9e61d35d3ca5c6e1e3922e7a9d86b6e432b6667f20bdff652e1e019f7aaf6b`.
The nested fitter retains its historical algorithm-kind name containing
`original_planner`; the outer report's explicit recorded-policy source,
compiled model, trace hashes and dual comparisons define actual provenance.

### Why the new hand fit still penetrates the floor

The legacy source scene includes `g1_29dof_old.xml`. Its rubber-hand meshes
are **visual-only**, with `contype=0`, `conaffinity=0` and no explicit contact
pairs. The native23 model's rubber-hand collision meshes are enabled. The recorded
source hand-crawl puts its visually rendered hands below the floor on 472/606
frames, reaching 145.947 mm below it. Its apparently modest active-contact
penetration therefore does not qualify the full physical hand geometry.

At frame 191, the source's 18 cm-offset virtual hand points are 87.824 and
146.754 mm below the floor. The native23 fit tracks those virtual points and
puts the right physical hand 191.225 mm through the floor. The common SONIC
VR offsets are coordinate conventions, not physical contact landmarks. They
remain unchanged; disabling native23 hand collisions would hide the problem,
not solve it. Dance has no below-floor hand points/meshes in this audit, so
this crawl-specific finding does not explain its remaining ankle/contact
tracking failures or the physical motors-off incident.

Evidence: `original29_hand_contact_audit_20260906_v1/report.json`, SHA256
`3809e6d0550cfc1ad318c1689c9b41ae496931795f6cf6edc0aae826a89dcb3b`.
The source comparison next needs physically meaningful hand-contact geometry
and contact landmarks, separately from unchanged teleop coordinates. Contact,
force and controller-constrained native23 fitting/training are still required.

### New paired controller replays still reject dance and return

The unchanged, correctly paired baseline100 encoder `3806b2b6...` and decoder
`90e27f37...` were tested against both new complete references plus the
historical-measured-start dance. All three fixed-envelope cases fail an empty
effort/position/slew intersection: hand 48/595, dance 47/535, historical-start
dance 17/535 completed transitions. Startup still passes 250/250 transitions;
return fails at 0/250, requiring 14.8714 rad/s instantaneous right-ankle target
slew against the unchanged 5 rad/s bound. The failed partial physics substeps
remain recorded. No guard was relaxed and no controller was deployed.

Elbow remains explicitly unavailable, not silently removed into an all-pass
suite. The five unchanged PICO clips were not rerun here. Historical posture
is not fresh robot state, and the standing compatibility actor is not a native
Unitree FSM handoff. No corrected matched-budget original-v14 comparison,
safe physical standing return or live PICO input qualification exists yet.

Evidence: `original29_recorded_envelope_20260906_v1/suite_summary.json`, SHA256
`3c9d22c3ebb878eef79372b5bcc1f93f619b5d073962a44671ea5d0e363373b6`.

The broad regression suite passes **398 tests**, no skips, in 163.67 seconds;
its two legacy asset-dependent modules use the original repository's assets.
All seven new Python source/test files pass Ruff. The standalone C++ arithmetic
witness compiles and passes its numerical tests. This is code verification,
not motion or hardware readiness. All 87 unique bound evidence/input files
across the four new reports were rehashed with zero mismatches. Existing dirty
hardware work is preserved.

## 2026-09-06 continuation: original-planner lineage and whole-clip task fitting

**NOT ready for physical dance, standing return or live full-body teleop.**
This work is offline. No robot connection, mode command, controller/interlock
edit, training run or policy promotion occurred. The physical bit-30 motors-off
cause remains unknown; these reference findings do not diagnose that event.

### The old references were not all original planner choreography

The independent, full-length lineage audit found that the legacy happy-dance
and elbow-crawl native23 references contain previous native23 controller
rollout states. Their provenance chain is intact, but that does not make their
paths identical to the original released planner. The hand-crawl reference
was already retargeted directly from the planner. All 1,778 compared frames
remain on the original 30-to-50-Hz time grid, with no time warp or cropping.

| Original clip | Legacy root error, translation aligned | Conservative error lower bound after allowed repair, including any constant yaw |
|---|---:|---:|
| Hand crawl | 0.000000191 m | 0 m |
| Elbow crawl | 4.464152 m | 4.213283 m |
| Happy dance | 0.673864 m | 0.439568 m |

The lower bounds allow every frame's original +/-0.08 m XYZ correction box
plus the 2e-7 saved-path tolerance, independently of temporal/contact/force
constraints. Elbow and dance cannot reach the existing provisional 0.25 m
pelvis-error screen by repairing those legacy paths inside those boxes.
This is **not** a proof that native23 physical dancing is impossible.
The original happy planner itself travels 8.975195 m horizontally; calling all
of that travel controller drift would also be wrong.

Evidence: `artifacts/g1_true23_frozen_lora/original_sonic_lineage_20260906_v1/report.json`
(SHA256 `1ce5bf318179b3c01098c3a459d628051a21cb51089f2aa8360b51f866bd3fe7`).
Twenty-two source/evidence files were hash-bound and rechecked. The five PICO
entries are explicitly outside the original-SONIC-planner lineage comparison.

The old parity summarizer also treated full-length survival as choreography
parity. Its new schema records `completion_only_match` separately and keeps
`parity.achieved=false`. The downstream candidate summarizer rejects the old
affirmative schema and cannot select a default from completion alone. Existing
historical JSON artifacts are preserved, not rewritten into successful results.
The lineage/parity regression suite brings the preceding broad suite to
**350 passing tests**, no skips, in 138.96 seconds; all six changed/added Python
files pass Ruff.

### Force V3 was stopped, not completed

`contact_force_trajectory_20260906_v3` was intentionally interrupted after the
lineage finding. Upright/standing completed their conditional force screens.
Crouch accepted ten whole-path updates; its last terminal-only residual was
4.458876462773899 squared normalized force error, maximum mixed generalized
force residual 44.28652189131918, and one geometry-violating frame. That partial
crouch path was not saved. `interrupted.json` records exit 1 and the reason.
There is no complete eight-clip V3 result, no final crouch witness and no V3
paired replay. The earlier completed force-V1 evidence remains historical.

### Rebuilding from the actual original planner

The first unretimed task-space rebuild hit the existing lower/root
initializer's invalid-frame check on hand crawl and wrote no native clip.
Its exact failure is retained in
`original_planner_task_space_20260906_v1/failed.json`. That initialization
failure is not physical infeasibility, and no slower timing was substituted.

An explicitly diagnostic initializer then let all 23 joints participate in
local task-space IK, deferring whole-path contact/force qualification. It
generated all three complete clips: 606 hand-crawl, 606 elbow-crawl and 546
happy-dance frames. Serialized FK and unchanged 5 rad/s, 80 rad/s^2 joint
limits pass. The new source-relative root errors are below 0.0000004 m, but
**all three task-space kinematic gates fail**. This material is not an accepted
teacher. Evidence: `original_planner_seed_20260906_v1/report.json`, SHA256
`02ed4817435b624cc68dd7d57a497c3420648d482f3708fe8e1bd456a85b0608`.

A separate whole-horizon nearest-joint projection tests the sequential
initializer's timing error without changing root pose, tempo or the limits.
It also writes all three clips and preserves the five PICO entries unchanged
in an explicitly diagnostic eight-clip manifest.

| Original clip | Local task-space seed: worst foot error | Whole-horizon joint projection: worst foot error |
|---|---:|---:|
| Hand crawl | 8.210 mm | 9.275 mm |
| Elbow crawl | 34.701 mm | 15.664 mm |
| Happy dance | 109.000 mm | 47.731 mm |

Every value still exceeds the unchanged 5 mm foot screen. Joint projection
does not compensate missing waist/wrist motion. Independent full-eight-clip
collision checks also fail: original hand/elbow/dance mesh floor overlaps
reach 115.271, 83.414 and 21.781 mm respectively. The training collision model
also fails. No floor geometry, physical limit or acceptance criterion changed.

Evidence under `artifacts/g1_true23_frozen_lora/`:

- `original_planner_projected_seed_20260906_v1/report.json`, SHA256
  `20fca9b2253fbb29585589c1445ad4258d692d84572eb6e03e20076f79c7035f`.
- `original_planner_projected_geometry_20260906_v1/report.json`, SHA256
  `021b438cb1d7eb1bb05c388c3dd0ba91a45ceafa421251bb7c3910160df8815a`.

### Bounded pelvis-attitude task fit and complete independent checks

New additive `g1_true23_original_task_trajectory.py` fits the original29
task poses with all 23 actual joints plus reference pelvis XYZ and attitude.
Unlike previous fixed-attitude reference optimizers, it explicitly permits a
bounded pelvis tilt to compensate the absent waist roll/pitch. It adds no
physical joints and no floating-base force actuators. All original frames,
source targets and 50-Hz timing remain present.

The fixed experimental attitude bound is L1 <= 0.45 rad, which also bounds
geodesic angle below the existing 0.5 rad fidelity screen. Root-rotation
coordinate derivatives are bounded at 1.5 rad/s and 12 rad/s^2; these are
provisional **rotation-vector coordinate** constraints, not measured physical
angular-acceleration limits. Existing root XYZ +/-0.08 m, native joint
correction +/-0.6 rad, safe target/action bounds, 5 rad/s and 80 rad/s^2 stay
in force. The QP retains a serialization reserve and independently audits all
original rows; nonlinear line search checks the actual task error and path.

This is a weighted geometric objective, not a lexicographic contact solver or
a certificate that the 5 mm foot screen passed. Collision, support/force,
constrained-controller tracking and safe transitions remain independent gates.
All teacher/deployment/hardware flags remain false. Nineteen focused tests
pass, including exact SE(3) Jacobians against finite differences, all 29
reference-variable columns, frame isolation, constrained missing-waist
compensation, float32 reconstruction, strict QP rejection and rejecting a
forged increment solution using the original absolute constraint rows.
The final broad suite passes **369 tests**, no skips, in 180.42 seconds.
Ruff passes for all eight changed/added Python source and test files.

The first all-three attempt (`original_planner_se3_fit_20260906_v1`) rejected
every first QP on the unchanged `1e-8` independent row tolerance. Its source
was archived byte-exactly before numerical reformulation; see its
`source_snapshot/manifest.json`. The next implementation solves the **same**
QP in current-increment coordinates and scales the entire objective by frame
count, preserving all relative weights. Both increment and original absolute
rows are independently checked at `1e-8`; no tolerance was increased.

The complete V2 attempt, `original_planner_se3_fit_20260906_v2`, keeps all three
original clips. Hand crawl accepted 24 updates; elbow crawl accepted 20 then
stalled; dance accepted five then rejected iteration six's `1.87228e-8` row
residual. Saved outputs pass independent temporal/FK reconstruction, but
neither optimizer status nor file generation is qualification.

| Original clip | Final worst foot error | Final worst hand-position error | Original-planner root-path error, translation aligned |
|---|---:|---:|---:|
| Hand crawl | 0.980 mm | 62.301 mm | 0.054068 m |
| Elbow crawl | 0.844 mm | 62.431 mm | 0.032498 m |
| Happy dance | 8.501 mm | 92.360 mm | 0.074245 m |

The two crawl foot screens pass; dance still fails the 5 mm screen. Actual
original-relative pelvis rotation maxima are 0.424714, 0.284156 and 0.381015
rad respectively. Original task poses and motion timing were not replaced
with native23 controller rollout states.

Full independent geometry and conditional inverse-force checks cover all
**6,035 frames and 5,955 causal packets**, including all five unchanged
original PICO inputs. Those PICO inputs are the raw original-manifest clips,
**not** the later contact-restored PICO variants; their zero force passes
below are not a regression of the previously restored upright/standing paths.

| Clip | Floor-overlap frames, mesh / training | Conditional force-pass frames, mesh / training |
|---|---:|---:|
| PICO upright, 1,024 frames | 0 / 0 | 0 / 0 |
| PICO standing, 1,024 | 0 / 0 | 0 / 0 |
| PICO crouch, 1,024 | 957 / 957 | 0 / 0 |
| PICO walk001, 695 | 49 / 63 | 0 / 0 |
| PICO walk010, 510 | 32 / 43 | 0 / 0 |
| Original hand crawl, 606 | 549 / 554 | 252 / 252 |
| Original elbow crawl, 606 | 441 / 476 | 24 / 39 |
| Original happy dance, 546 | 270 / 282 | 6 / 6 |

The unchanged inverse-force limit is `0.2375 * effort`, with all floating-base
equations and both compiled collision models. Its conditional successes do
not prove contact complementarity or control tracking. Hand-crawl floor
penetration still reaches 126.223 mm in the training model; better task-pose
matching does not imply valid contact geometry. None of these clips is an
accepted teacher. The five original PICO inputs need their own contact/force
restoration lineage preserved when constructing a future combined corpus.

Fresh paired-controller simulations also ran every full requested clip plus
historical-measured-start happy dance. This uses the existing fixed V2 case
commands, correct encoder `3806b2b6...` / decoder `90e27f37...` pair, constrained
effort, current-observation timing and 5 rad/s slew. The new diagnostic
manifest is explicitly validated, not relabelled as an accepted teacher.
All nine cases fail; none establishes dance, return or live PICO readiness:
upright 2/1,013; standing 2/1,013; crouch 3/1,013; walk001 1/684; walk010 1/499;
hand 2/595; elbow 2/595; happy 2/535; historical-measured happy 16/535.
Every failure is an empty effort/position/slew target intersection. Historical
measured startup passes 250/250 transitions, but return fails at 0/250; the
right ankle would need 11.6353 rad/s instantaneous target slew, beyond the
unchanged 5 rad/s bound. No runtime limit was increased. This is not a native
Unitree FSM handoff or fresh physical telemetry.

Evidence under `artifacts/g1_true23_frozen_lora/`:

- `original_planner_se3_fit_20260906_v2/report.json`, SHA256
  `0161dc17155c49bd9d86eb20cdd93d4f66da1b7c9d90c443101f4fa64d2693ff`.
- `original_planner_se3_screen_20260906_v2/report.json`, SHA256
  `5b62c6bc41330440fa0944aa8a28ff1cbc6079065e2c5b6e6f23e8f9f0b23f7f`.
- `original_planner_se3_envelope_20260906_v2/suite_summary.json`, SHA256
  `b95b4a3f0ddd5c72033c221bc1a792ca97397a7bfca63ed39c3347e9ff51f4b5`.

No corrected full-weight-v14 matched-budget comparison exists yet. Further
work must retain original-source fidelity while making contacts and forces
valid; purely geometric fitting must not be promoted into a physical teacher.

### Original29 source audit: nominal planner fidelity is not stock-policy parity

A further full-length source check confirms the raw planner poses themselves
are not valid rigid-floor references. In both the original29 retarget model
and the current stock29 scene, hand crawl has 549 overlapping frames with
115.250 mm maximum penetration; elbow has 442 with 83.343 mm; happy dance has
151 with 20.835 mm. The worst crawl penetration is at the knee/hip geometry,
not a missing wrist. Thus these source defects cannot be attributed solely to
the native23 embodiment or fixed by faithfully copying nominal foot poses.

The hash-matched archived original29 controller trajectories are a separate
comparison. Their translation-aligned planner root errors are 2.176930 m for
hand crawl and 0.329613 m for dance. These are all stored post-control samples
matched to the same-index planner command, with the one-control-interval
sampling offset explicitly disclosed; no phase shift, cropping or time warp
was used. Both exceed the provisional 0.25 m nominal-planner root screen.
Therefore that screen is **not an established stock-SONIC parity criterion**;
the lineage lower bounds above apply to the nominal planner path, not proof
that stock-policy choreography cannot be matched. New qualification must
report both original29-policy-relative motion and nominal-planner fidelity.

The checked-out stock joint-mode encoder constructs joint/velocity/orientation
inputs but does not feed planner root XYZ to the policy. This is consistent
with treating nominal translation and executed stock motion separately, not
assuming exact nominal root tracking. The archived original29 trajectories
also reach joint speeds 12.972 and 15.771 rad/s and have soft-contact overlap
up to 15.803 and 8.241 mm. They are not automatically valid native23 references
under the unchanged limits. No elbow original29 trajectory exists in that
archived two-clip suite, and the archive does not bind the full compiled-model
and source closure. A new properly bound stock baseline remains required.

Evidence: `original29_source_screen_20260906_v1/report.json`, SHA256
`90515ae218c7885eac2ad851781e5e9357a10ef02b132fe982ffe1bc67816628`.
The scene wrapper and archived NPZ/report hashes were checked; this was
read-only FK/contact inspection, not a fresh original29 policy execution.

The upstream [transfer project](https://sonic-agibot-x2.github.io/sonic-transfer/)
also now describes unresolved physical torque saturation despite successful
simulation. That is a separate X2 observation, not an explanation of this G1's
damping, and reinforces why simulation success alone cannot qualify hardware.

## 2026-09-06 continuation: full-path contact-patch force optimizer

**NOT ready for physical dance, standing return or live full-body teleop.**
This continuation remains offline. No robot connection, mode command, control
loop edit, policy promotion or safety-limit/interlock change. The physical
bit-30 motors-off cause remains unverified.

The full-path solver now has an explicit Clarabel 0.11.1 backend, strict
`Solved` status plus independent original-row audit at `1e-8`, a correction
cost centered on the current iterate, bounded trust retries, and hard local
candidate-support patches. All frames have independent root-XYZ/23-joint
variables; no stationary tail is tied and no entry motion is removed. Original
position/correction boxes, root orientations, 50 Hz timing, velocity,
acceleration and initial velocity remain unchanged. Both compiled models share
the same path but retain independent contact-force/torque variables. The
optimizer requests 98% of the already constrained `0.2375 * effort` limits;
this is a 2% algebraic screening reserve, not robust-control headroom proof.

The new implementation is additive (`g1_true23_box_qp.py`,
`g1_true23_contact_force_optimizer.py`, `refine_g1_true23_contact_forces.py`).
Historical force/contact utility and CLI bytes remain unchanged: all 161 files
in the preceding completed force-V1 comparison were independently rehashed
with zero changes. Candidate output is saved and reopened before independent
temporal/FK, every-frame inverse-dynamics LP and causal-packet checks.

### Interrupted V2 attempt and guard correction

`contact_force_trajectory_20260906_v2` is an **interrupted partial experiment**,
not a completed corpus. Upright/standing each completed 1,024/1,024 conditional
force/headroom frames in both models. Crouch accepted two full-path updates
before the agent stopped the offline process to correct an overly restrictive
nonlinear guard check. Last terminal-only crouch values: normalized squared
residual 10.412302965622153, maximum mixed generalized-force residual
50.740144294183324, two geometry-violating frames. That partial crouch candidate
was not serialized; it is not a witness or full-motion result. Exit 1 was an
intentional SIGINT during iteration-three linearization, not a completed test.
`interrupted.json` records this distinction; five byte-exact source snapshots
were verified before editing. The pre-correction broad suite passed 329 tests.

The linear patch ceiling is 1.95 mm (50 micrometers inside the unchanged 2 mm
candidate band). Nonlinear curvature may consume that reserve, but must remain
0.2 micrometers inside the outer band. The first attempt incorrectly required
almost the entire reserve to remain unused. The corrected check retains actual
contact recomputation and the independent force/geometry audits. Focused tests
after this correction: 33 passed, no skips, including rejecting exhausted
guard, reduced solver status and forged successful solver results.
The final broad suite on the corrected code passed **332 tests**, no skips,
in 135.88 seconds. Ruff checks passed for all five added Python files.

### Historical full-corpus V3 attempt (now interrupted; see above)

`artifacts/g1_true23_frozen_lora/contact_force_trajectory_20260906_v3` was started
fresh for all eight clips, up to 64 nonlinear iterations and 200 QP iterations.
`started.json` binds source/input bytes before work begins and is explicitly
not a completion record. This section must not be interpreted as a finished
full-corpus or replay result. A completed `report.json`, all eight final saved
references, independent force LPs and changed-reference full paired replays
remain required. No training or hardware readiness follows from optimizer
status, and original-full-weight-v14 corrected matched-budget comparison is
still outstanding.

The first V3 crouch update is a measured optimizer result, not a saved clip:
Clarabel `Solved`, independent original-row violation `2.2950391587173158e-15`,
full line-search step at trust fraction `1/16`, nonlinear guard consumption
31.799618314815254 micrometers (below 49.8 micrometers). Normalized squared
force residual fell to 9.390167177444004; maximum residual remained
52.47102125020848 and six geometry-violating frames remained. Thus the reserve
correction permits a finite full-path update but does not establish feasibility.

## 2026-09-06 continuation: force-path restoration and a conditional single-pose witness

**Physical dance, standing return and live full-body teleop remain NOT ready.**
This continuation is offline-only. No robot connection, command, mode change,
hardware-controller edit, policy training/promotion or limit/interlock
relaxation has occurred. The previous pass's rejected contact-only candidates
remain unaccepted. The physical bit-30 motors-off cause is still unknown.

### Force-aware implementation

New `g1_true23_force_trajectory.py` computes whole-path required generalized
forces and their sparse derivatives. It retains all 23 joint coordinates,
root-XYZ offset variables, original root orientation, 50 Hz sample timing and
all six unactuated floating-base equations. Independent central/one-sided pose
differences, not archived velocity arrays, couple adjacent frames and endpoints.
Required force matches independently calculated `M @ qacc + bias - passive`.

The derivative model is a private copy with constraints and `invdiscrete`
disabled, because candidate contact forces are explicit optimization unknowns.
[MuJoCo's continuous-time inverse-dynamics derivative API](https://mujoco.readthedocs.io/en/3.5.0/APIreference/APIfunctions.html#mjd-inversefd)
supplies transposed position/velocity derivatives; acceleration derivatives use
the exact mass matrix. Original replay/training models are not modified. Their
hashes, private derivative-model hashes and candidate-contact-model hashes are
recorded separately.

Both collision models share one trajectory but have separate unilateral
contact-cone force variables, bounded actuator torques and explicitly optimistic
bounded friction-loss assistance. No actuator supplies floating-base force.
A bounded least-squares force seed is not a feasibility certificate. Local
contact-load derivatives hold body-attached points and world cone directions
fixed; they do not predict closest-feature changes or prove sticking contact.
Every proposed path recomputes actual candidate contacts and force fits.

New `g1_true23_force_restoration.py` solves a scaled, whole-path QP coupling
these force equations with the existing contact, position, velocity and
acceleration constraints. Absolute normalized force residuals and contact
slacks permit restoration, but neither is accepted as final force support or
contact feasibility. A synthetic regression demonstrates why a squared force
penalty can retain an avoidable nonzero force mismatch; the absolute penalty
removes it in that fixture. This is not a general convergence claim.

Only original correction boxes and the same 0.95 × quarter-effort limits are
used. Existing deployed PD target-position/slew dynamics are not replaced by
inverse-force equations. Final force balance, no-slip/contact complementarity,
full policy tracking, standing return and hardware qualification remain
distinct requirements.

New `refine_g1_true23_reference_forces.py` processes every supplied clip into a
fresh, explicitly unaccepted corpus. It rejects changed prior source pins,
validates the completed input corpus, rebuilds float32 FK/derivative arrays and
all causal packets, then runs an independent full-frame force LP on both models
for every output. Failures remain represented. No command/packet is sent.

### Complete full-corpus verification

**299 tests pass, no skips**, including 22 focused force-path tests and seven
candidate-support-patch tests. Full lint passes on the six new Python files.
The actual eight-clip restoration run completes with exit 0, all **6,035
supplied frames** and **5,955 rebuilt/validated causal packets**, original sample
timing/root orientation and all 23 joint references. Numeric thread counts are
fixed at one. Eight restoration iterations maximum, 30,000 QP iterations,
2e-7 path audit tolerance and 1e-5 generalized-force tolerance are unchanged.

Artifact scope: `artifacts/g1_true23_frozen_lora/force_trajectory_20260906_v1/`.
The final manifest/report is complete. Upright and standing retain their
existing conditional contact/force passes. The other six first QPs all reach
30,000 iterations and are rejected without an accepted update:

| Clip | QP solve time (s) | Primal residual | Dual residual |
|---|---:|---:|---:|
| PICO crouch | 1,028.014 | 2.3424e-5 | 0.01352 |
| PICO walk 001 | 595.462 | 1.6746e-4 | 0.56770 |
| PICO walk 010 | 498.068 | 1.5081e-4 | 0.21693 |
| SONIC hand crawl | 567.056 | 4.9887e-4 | 0.14762 |
| SONIC elbow crawl | 557.585 | 2.7355e-3 | 0.43403 |
| SONIC happy dance | 533.241 | 6.5449e-4 | 0.06516 |

These numerical failures are not infeasibility proofs. **All eight output NPZ
files are byte-identical to the prior contact V2 candidates.** Both-model full
force audits are unchanged: conditional passing counts (mesh/training) are
crouch 1/2 of 1,024, walk 001 2/1 of 695, walk 010 1/1 of 510, hand crawl
57/57 of 606, elbow crawl 79/80 of 606, and dance 65/65 of 546. No clip is
removed, no reference is accepted and no policy is trained or selected.

No duplicate closed-loop run is claimed on these identical inputs. The
independently rebound prior nine complete requested replays still fail. Dance
remains 51/535 reference-start and 18/535 historical-posture transitions, with
0/2,500 return physics steps. This is reused simulator evidence, not a new
run or a physical Unitree mode transfer.

`force_trajectory_screen_20260906_v1/comparison.json` independently rechecks
**161 files**, including the unchanged references, prior paired replay evidence
and the separate stationary witness. SHA-256:
`340f36fffc8c0dd39ea5d9d605cfbb746a1f1ba237f469ab1c0d55f79e915916`.
Full force-refinement report SHA-256:
`695ac7a77240b617754b8a5930d6bedde393df7fc636e1d12fb28a7f9fcb615c`.

### Numerical and contact-patch diagnostics (not full-clip qualification)

The full crouch QP reaches 30,000 iterations after 1,028.0 seconds, with primal
residual 2.3424e-5 and dual residual 0.01352. A three-frame repetition of source
frame 512 reproduces the numerical failure. It is explicitly a stationary
fixture, not a shortened motion test or a replacement reference.

[Clarabel 0.11.1](https://pypi.org/project/clarabel/0.11.1/) was installed with
`--no-deps` for isolated numerical diagnostics only. MuJoCo, NumPy, SciPy and
OSQP versions remain unchanged. Its 22-iteration solve of the same small QP
passes the original independent 1e-8 linear audit. The actual nonlinear step
still fails: the full trial loses all candidate contacts in both models, while
the linearized wrench equations retain the old contacts. The independent L1
force-fit diagnostic also worsens, so that first rejection is not merely a
least-squares-versus-absolute-merit artifact. No solver status is promoted.

New `g1_true23_contact_patch.py` exposes conservative body-attached candidate
surface-point values and all-26-coordinate derivatives. It correctly separates
[MuJoCo's contact midpoint from the material surface point](https://mujoco.readthedocs.io/en/3.5.0/APIreference/APItypes.html#mjcontact).
Every current cone candidate is retained inside a stricter 1.95 mm band, not
only those carrying the previous force seed. This is a local optimization
hypothesis, not a contact-persistence, closest-feature or no-slip proof. The
full-corpus V1 implementation was not modified or hot-patched during its run.

Isolated Clarabel probes with these additional hard trial rows make four
accepted steps before nonlinear stalling. Reducing trust steps by 16 makes
16 accepted steps before an `AlmostSolved` result, which is rejected. Tying
the repeated pose variables exactly removes that fixture's near-zero temporal
degeneracy and allows 58 accepted steps. Maximum force-fit residual falls from
13.7112 to 0.000628988, but the serialized pose still exceeds the mesh-model
effort limit; only the training-model static LP passes. That candidate is
rejected. A separate minimum-step continuation centers the QP pose cost on the
current iterate rather than pulling toward the old pose while restoring force
feasibility. It converges in two steps with maximum force-fit residual
5.5334e-9. This is an isolated diagnostic override, not a changed production
optimizer or a full-clip result.

A closed-form curved-force regression separates objective-induced tangent
drift from solver accuracy: an old-pose-centered linear optimum can fail every
nonlinear trial while a current-pose-centered minimum correction succeeds.
OSQP also failed convergence on the tiny numerical version of that fixture;
the closed-form test does **not** claim that its numerical solver passed.

An independent disk-round-trip audit rechecks 20 evidence files and verifies
original correction bounds, serialized contact clearance/support bands, FK,
root orientation and exactly zero pose-derived velocity/acceleration. All 16
repetitions of that single pose have a conditional inverse-force solution on
both models. Mesh peak effort ratio is 0.999999982, versus 0.937458101 for the
training model: **the mesh witness has essentially no effort headroom**. It is
not a robust reference, teacher, closed-loop balance test or physical result.
Candidate contacts and friction assistance remain optimistic hypotheses.

Witness evidence:
`force_trajectory_minimum_step_static_crouch_20260906_v1/serialized_witness_audit.json`
under the frozen-LoRA artifact root, SHA-256
`3e8919de931de2c02e457b85d9199f6c14d9b83645d651d587e40e8cf4b2799f`.

The original crouch source is not constant over the complete clip: it has 72
distinct poses and initial projection velocity up to 3.17468 rad/s. A stationary
fixture must therefore not replace its full 1,024-frame transition/hold path.

### Next qualification work

Transfer the validated minimum-step/support-patch ideas to the actual whole-path
solver with an explicitly pinned, independently audited alternative backend.
Establish usable effort headroom and preserve every nonstationary phase, then
recheck all eight full references and closed-loop fidelity before retraining or
policy selection. Live PICO retargeting, full standing return, a corrected
matched-budget original-v14 comparison and the physical bit-30 diagnosis remain
separate unfinished requirements. Offline whole-horizon reference optimization
is not a causal live-input algorithm and does not establish exact 29-to-23 parity.

## 2026-09-06 continuation: coupled contact refinement improves geometry, not dance readiness

**Physical dance, standing return and live full-body teleop remain NOT ready.**
This pass changes offline reference refinement only. No robot connection,
command, mode change, controller edit, training, policy promotion or limit/
interlock relaxation occurred. Pre-existing dirty hardware work is untouched.
The physical bit-30 motors-off cause is still unknown.

### Implemented

New `refine_g1_true23_stance_contacts.py` and
`g1_true23_contact_trajectory.py` jointly refine whole-path contact geometry and
position/derivative bounds. The previous framewise soft floor objective followed
by a separate temporal projection could leave or reintroduce floor penetration.
This new sequential convex restoration independently re-evaluates every actual
collider distance on both the retarget mesh and actual training capsule models.

Each original inferred support-body hypothesis remains unchanged. Every
floor-colliding shape must clear the floor by 0.2 mm; each hypothesized support
body must retain at least one collider within 2 mm. These are geometry
constraints, not evidence of a load-bearing contact patch, COM balance, no-slip
support or realizable contact forces. Empty inferred support phases remain
empty; no artificial all-clip double-foot schedule is substituted.

Variables are root XYZ offsets and all 23 joint coordinates at all original
50 Hz samples. Original root orientation and timing remain fixed. The previous
correction bounds remain: root offsets within 0.08 m per axis, joints within
0.6 rad of the original and the existing safe/hard-margin bounds, joint
velocity/acceleration at 5 rad/s and 80 rad/s², root-offset derivatives at
0.75 m/s and 6 m/s². These reference limits do not expand hardware authority.

Sparse whole-horizon QPs use the independently pinned optional dependency
`gear_sonic[contact_retarget]` / `osqp==1.0.5`. Temporary nonnegative contact
slack permits numerical restoration; **no slack is permitted in final
acceptance**. Solved status alone is insufficient: an independent linear-bound
audit, nonlinear distance audit, float32 serialization audit and full-path
derivative/FK audits must pass. Root quaternions are compared as SO(3) rotations,
not component equality; sign/normalization differences are accepted only within
the explicit 2e-7 rad serialization tolerance. All causal packet terms are
rebuilt from the new arrays. Neither packets nor robot commands are published.

Actual validation uses the existing WSL MJLab environment: Python 3.11.15,
MuJoCo 3.5.0, NumPy 2.3.4, SciPy 1.16.2 and OSQP 1.0.5. Only OSQP was added,
without dependency upgrades/downgrades. The new extra declares the additional
solver dependency, not a complete MJLab environment. The legacy package's
NumPy/SciPy and `[sim]` pins differ from this recorded audit runtime; do not
replace the validated environment with those older pins and assume parity.

The sparse trust-box row reduction removes only constraints redundant for that
linearized box; every nonlinear collider is still audited. A stall or the
20-iteration restoration limit produces a failed candidate, not an infeasibility
theorem and not a dropped clip. QP numerical budget is 100,000 iterations with
unchanged 1e-8 solver tolerances and 2e-7 independent acceptance tolerance.

### Verification and complete corpus results

**270 tests pass, no skips**; all three new Python files pass full lint and the
optional dependency contract parses. Tests cover analytical contact Jacobians
against finite differences, all-26-coordinate temporal coupling, incompatible
two-model support bands, solver false-success rejection, retained original
frames/bounds and failure status. V1 prototype output is partial and is not a
completed corpus. V2 completed with exit 0, all **eight clips / 6,035 supplied
frames**, all 23 joints and **5,955 rebuilt/validated causal packets**. Every
serialized path passes its derivative and FK audits. No frame was dropped,
source sample rate changed, or candidate promoted.

The following counts combine clearance and inferred support-gap violations
across both collision models. They are **not floor-penetration counts**.
The old upright/standing failures were sub-4-micrometer clearance shortfalls,
not actual floor penetration. Distance acceptance tolerance is 0.0002 mm.

| Clip | Frames | Violating frames before | After | Worst remaining violation (mm) | Geometry pass |
|---|---:|---:|---:|---:|---|
| PICO upright | 1,024 | 1,024 | 0 | 0.000028 | Yes |
| PICO standing | 1,024 | 1,024 | 0 | 0.000053 | Yes |
| PICO crouch | 1,024 | 1,024 | 0 | 0.000003 | Yes |
| PICO walk 001 | 695 | 280 | 2 | 0.203269 | No |
| PICO walk 010 | 510 | 129 | 0 | 0.000163 | Yes |
| SONIC hand crawl | 606 | 468 | 70 | 11.019036 | No |
| SONIC elbow crawl | 606 | 514 | 514 | 235.078837 | No |
| SONIC happy dance | 546 | 444 | 1 | 0.000426 | No |

Walk 001 reaches the 20-iteration restoration limit; its largest residual is
a right-foot support gap, not floor penetration. Hand crawl's third QP reports
`solved inaccurate` after 100,000 iterations; elbow crawl's first QP reaches
that numerical limit. Both are rejected, not treated as proof that no feasible
trajectory exists. Dance stalls after 14 accepted updates, leaving a 0.426 µm
support-gap violation at frame 1 on the training model's left foot. Tolerance
is not widened to pass it. Dance has **zero penetrating frames in both models**,
versus 382 mesh / 391 capsule frames before; it still is not an accepted motion.
Hand crawl retains four penetrating training-model frames; elbow crawl retains
462 mesh / 376 capsule penetrating frames.

### Geometry is not force balance

The independent full-reference inverse-dynamics audit derives unsmoothed
velocities/accelerations from all poses and uses the same 2 mm candidate-contact
gap, both models, and unchanged 0.95 × quarter-effort limits. It does not prove
contact complementarity, no-slip kinematics or controller tracking. Counts below
are frames with a conditional force solution within those supplied limits,
shown as mesh / training capsules.

| Clip | Previous stance V2 | Contact-refined V2 | Total frames |
|---|---:|---:|---:|
| PICO upright | 1,024 / 1,024 | 1,024 / 1,024 | 1,024 |
| PICO standing | 1,024 / 1,024 | 1,024 / 1,024 | 1,024 |
| PICO crouch | 1 / 3 | 1 / 2 | 1,024 |
| PICO walk 001 | 2 / 1 | 2 / 1 | 695 |
| PICO walk 010 | 0 / 0 | 1 / 1 | 510 |
| SONIC hand crawl | 56 / 56 | 57 / 57 | 606 |
| SONIC elbow crawl | 79 / 80 | 79 / 80 | 606 |
| SONIC happy dance | 65 / 65 | 65 / 65 | 546 |

Only the two upright/standing reference clips pass at every frame in both
models. Crouch illustrates the missing constraint: frame 512 has exactly zero
reference velocity/acceleration, but its COM violates a candidate support-hull
halfspace by 35.75 mm on the mesh and 13.70 mm on the training model. The COM is
also behind every candidate contact in world X. Both force LPs reject it.
Restoring floor clearance does not put the COM over a usable support region.
These are supplied-model diagnostics, not measurements of the physical robot.

### Identical-policy full replay rejects the candidate

Both reference sets use the same actual baseline-100 checkpoint pair:
encoder `3806b2b6...`, decoder `90e27f37...`. Gains, stateful applied-target
feedback, current-state observation timing, fraction 1, modeled 35 Nm ankles,
quarter-effort/95% projection and 5 rad/s target slew are identical. No training
occurred on the new candidates. Counts are completed 50 Hz transitions before
the unchanged effort/position/slew intersection becomes empty.

| Full requested motion | Requested | Previous stance V2 | Contact-refined V2 |
|---|---:|---:|---:|
| PICO upright | 1,013 | 48 | 48 |
| PICO standing | 1,013 | 38 | 38 |
| PICO crouch | 1,013 | 27 | 27 |
| PICO walk 001 | 684 | 17 | 17 |
| PICO walk 010 | 499 | 24 | 11 |
| SONIC hand crawl | 595 | 43 | 44 |
| SONIC elbow crawl | 595 | 52 | 52 |
| SONIC happy dance | 535 | 49 | 51 |
| Happy dance, recorded posture + 5 s standing | 535 | 70 | 18 |

**All nine new replays fail full-motion fidelity and lifecycle qualification.**
Reference-start dance reaches 512 physics steps; its maximum joint RMSE worsens
from 0.4106 to 0.4408 rad. Recorded-posture dance regresses from 707 to 189
physics steps. Its smaller truncated-prefix RMSE does not establish better
full-motion fidelity. Both recorded-posture standing startups pass 250/250
transitions / 2,500 physics steps, but both returns complete **zero physics
steps**. The new return requires instantaneous right-ankle target slew
5.07192 rad/s versus the unchanged 5 rad/s bound. Historical-posture simulation
is not fresh robot state and the compatibility standing actor is not a Unitree
FSM handoff. The physical damping fault remains unexplained.

### Evidence and next boundary

All artifact paths below are under `artifacts/g1_true23_frozen_lora/`:

- `contact_trajectory_20260906_v2/`: full manifest, all eight NPZs, per-clip
  numerical/serialized reports, input/source/model hashes and runtime versions.
- `contact_trajectory_support_20260906_v1/report.json`: both-model full-reference
  force audit; SHA256 `ecbba5b85fd28629eb41080afd561e9750095598644b6e5e551b9830db1d59f5`.
- `contact_trajectory_envelope_20260906_v1/`: all nine requested paired replays,
  exact commands, reports and actual 500 Hz traces; suite SHA256
  `8013406dc3ca8ec65f6c0ea46c1c6bce8fd81b4544d2b731efc17c52cce9dadd`.
- `contact_trajectory_screen_20260906_v1/comparison.json`: independently verified
  same-pair/same-actuation comparison, all counts and **120 rechecked files**;
  SHA256 `1670a3984047cf33adeff8bd3d7309231bba737c09348efc3b8a8bdfe5dab98a`.
  `build_comparison.py` reproduces the evidence audit in a fresh output scope.

The refinement entry point takes `--stance-dir` and a fresh `--output-dir`;
`audit_g1_true23_reference_support --reference-dynamics` consumes its manifest;
`evaluate_g1_true23_paired_envelope_suite` consumes that same manifest and the
baseline-100 paired export reports. These are offline-only diagnostics.

No candidate is selected. Next useful work must address realizable support/COM
and force constraints jointly with reference motion, then learned full-body
tracking and standing return. More geometric cleanup alone is insufficient.
This fixed-policy retargeting comparison is **not** a new matched-budget
original full-weight-v14 comparison, exact 29-to-23 motion parity, successful
live PICO input, or permission to run the physical robot.

## 2026-09-06 continuation: requested/projection cost; matched training rejects both policies

**Physical dance, standing return and live full-body teleop remain NOT ready.**
Actual training occurred in this pass, but no robot connection/command, mode
change, hardware-controller edit, limit/interlock relaxation or policy promotion
occurred. Pre-existing dirty hardware work remains untouched. The physical
bit-30 motors-off cause is still unknown.

### Implemented and tested

The existing target/reference and target-limit rewards see already constrained
targets. A policy can therefore request increasingly excessive targets while
receiving the same applied target and associated target-based cost. New opt-in
`--projection-penalty-weight` exposes that difference to PPO. Its default is
**zero**, and positive values require `native_support_stateful_v2`.

At every 2 ms substep, the action manager records squared requested-minus-
projected target error, normalized per joint by the unchanged native23 hardware
action scale. The reward averages all 23 joints and ten physics substeps per
50 Hz action. Per-component squared cost is capped at 100; nonfinite components
receive that cost while the existing action guard still terminates the episode.
Invalid substeps remain included in this reward accounting; this does not imply
nonzero effort was applied on a latched invalid substep. Selective environment
resets and each new policy action clear the appropriate accumulators.

All 15 previous rewards, gains, torque/position/slew limits, 930/267 policy
interfaces, frozen SONIC core, previous-applied-action feedback and termination
conditions remain unchanged. The new weight/contract is bound into preflight,
resolved training configuration and checkpoint lineage. A regression fixture
demonstrates identical projected controls for distinct requests while the new
reward distinguishes them; recording alone leaves controls/guard status exact.

The [upstream X2 project](https://sonic-agibot-x2.github.io/sonic-transfer/)
discusses actuator-limited requests hidden by simulation. That motivates this
training experiment; it does **not** establish the cause of this G1's hardware
fault or prove that this loss fixes it.

New `evaluate_g1_true23_paired_envelope_suite.py` evaluates every manifest clip
through the existing current-state, paired encoder/decoder envelope evaluator.
It requires matching export identities and full requested clip lengths, fixes
configured gains, fraction 1, modeled ankle rating 35 Nm, quarter-effort/95%
projection and 5 rad/s slew, and records actual 500 Hz trajectories. No legacy
decoder-only default, shortened-clip option, predictive filter or promotion
is used. Stance references retain their explicit unaccepted-candidate status.
Optional named measured-start cases require recorded health and the pinned
standing compatibility actor together, with five-second startup/return requests.

### Actual paired PPO runs

Both runs use the complete original eight-clip, 6,035-frame SONIC/PICO corpus,
seed 20260906, rank/alpha 8, learning rate 5e-6, 32 environments, 16 rollout steps,
five PPO epochs/eight minibatches and **100 updates / 51,200 transitions each**.
Neither was trained on stance-candidate references. Baseline weight is 0;
experimental weight is 2. Checkpoint-zero actor/adapter/critic tensors, optimizer
state and trainer state were compared directly and match bit-for-bit. Resolved
configs differ only in penalty enablement/weight; lineage differs only in those
payload fields and their derived hashes. All 18 adapter tensors changed in both
actual training runs; frozen-core contracts remained unchanged.

Final logged rolling episode lengths are 4.11 baseline / 4.15 penalty control
steps; last-ten logged means are 4.752 / 4.374. These are logger statistics, not
full-clip survival or a convergence claim. One seed and 100 updates are not a
generalization study. Strict materialization and both paired ONNX exports pass;
encoder FSQ tokens match exactly, decoder maximum absolute parity error is
2.6226e-6 / 1.5497e-6. Export correctness is not motion qualification.

### All full requested replays, no omitted clips

Counts below are completed 50 Hz transitions before target-intersection failure.
Each original or stance suite retains all eight clips and all 6,035 frames.

| Motion | Requested | Original baseline | Original penalty | Stance V2 baseline | Stance V2 penalty |
|---|---:|---:|---:|---:|---:|
| PICO upright | 1,013 | 2 | 2 | 48 | 46 |
| PICO standing | 1,013 | 2 | 2 | 38 | 37 |
| PICO crouch | 1,013 | 3 | 3 | 27 | 27 |
| PICO walk 001 | 684 | 1 | 1 | 17 | 16 |
| PICO walk 010 | 499 | 1 | 1 | 24 | 24 |
| SONIC hand crawl | 595 | 43 | 43 | 43 | 44 |
| SONIC elbow crawl | 595 | 51 | 55 | 52 | 53 |
| SONIC happy dance | 535 | 47 | 48 | 49 | 58 |

Two additional stance-dance cases start from the recorded posture after five
seconds of standing: baseline completes 70/535 transitions (707 physics steps),
penalty 75/535 (758 physics steps). Both standing acquisitions pass 250/250
transitions and 2,500 physics steps. **Both return requests complete zero physics
steps**, failing the same target-intersection constraint. Penalty's return
requires instantaneous left-ankle target slew 5.0722 rad/s versus the unchanged
5 rad/s limit. This is historical-posture simulation with the compatibility
standing actor, not fresh robot state or a Unitree FSM handoff.

**All 34 replays fail full-clip fidelity and lifecycle qualification. Neither
policy is selected.** Nine extra retargeted dance transitions are not a fix:
that case's maximum joint RMSE is 0.4106 baseline / 0.4619 penalty rad, and some
standing/walking cases regress. Original dance RMSE is 0.4081 / 0.4041 rad;
recorded-posture dance RMSE is 0.3922 / 0.4007 rad. Different terminated rollout
lengths also make full-prefix residual means unsuitable as matched-input or
complete-motion cost comparisons. The penalty remains disabled by default.

Evidence root: `artifacts/g1_true23_frozen_lora/projection_cost_20260906_v1/`.
`baseline100` and `penalty2_100` contain actual update-0/update-100 checkpoints,
training logs/lineage, strict paired exports and original/stance envelope suites.
`comparison.json` binds exact commands, direct tensor comparison, all results,
training scalar evidence and **185 independently rechecked input/source/output
files**. Its SHA256 is
`61d72c5b955f80a32e089af8b2ebb5b38d1eb4e2811fd2e468de47d482e7d152`.
The common encoder SHA256 remains `3806b2b6...`; baseline/penalty decoder hashes
are `90e27f37...` / `6e85051a...`, with complete identities in the report.

Verification: **255 tests pass, no skips**; all eight changed/new Python files
pass full lint. This is a matched reward ablation within frozen LoRA, **not** a
corrected matched-budget comparison with the original full-weight v14 trainer.
Native full-body closed-loop balance, realizable contact references, successful
full dance/acquisition/return and fresh operator-supervised evidence remain
required. No hardware limit may be inferred from these simulator experiments.

## 2026-09-06 continuation: fix stale replay observations; sampled training parity

**Physical dance, standing return and live full-body teleop remain NOT ready.**
No robot connection/command, mode change, hardware-controller edit, limit or
interlock relaxation, learning or policy promotion occurred. Existing dirty
hardware work remains untouched. The physical bit-30 motors-off cause remains
unknown; the simulation defects below are not established hardware causes.

### Implemented timing correction

`CleanTrue23MujocoController._policy_frame()` previously mixed current
`qpos/qvel` with the previous 2 ms physics step's derived body angular velocity.
The envelope evaluator also compared stale body positions against current
reference positions. [MuJoCo documents this post-step timing](https://mujoco.readthedocs.io/en/3.5.0/programming/simulation.html#simulation-loop).
MJLab refreshes derived state before policy observations; the CPU replay did not.

New `refresh_observation_kinematics()` calls only `mj_kinematics`, `mj_comPos`
and `mj_comVel`. Policy frames and envelope body-tracking metrics now use the
current post-integration state. It does not integrate, call the control callback,
solve contacts, change controls, or change `qacc_warmstart`. Regression tests
first reproduced the old gyro defect in upright and rotated free-body fixtures;
the corrected readings match an independently refreshed MuJoCo copy while
physics state, effort/constraint arrays and time remain byte-identical.
New envelope reports explicitly identify the observation/body-tracking timing.

### Actual training versus replay, not just duplicated controller equations

New `audit_g1_true23_training_replay_parity.py` samples four real phases from
each of the eight original corpus clips (32 total), with no reset rejection.
It invokes the actual MJLab V14/frozen-LoRA observation and native-support V2
action managers, then one actual 2 ms Warp step. Noise and reset perturbations
are explicitly zeroed for this deterministic diagnostic. No actor checkpoint
is trained or loaded into the training runner: the correct paired model-50
ONNX encoder/decoder evaluates both independently constructed observations.
Joint/state/actuator layout is checked, not inferred from dimensions alone.

On all 32 samples, semantic/proprioceptive observations, exact FSQ tokens,
raw actions, requested targets, applied targets and actuator torque agree
within their declared absolute tolerances. Maximum requested-target difference
is 2.3842e-7 rad; applied-target difference is exactly zero; torque difference
is 1.8229e-6 Nm. Warp versus CPU MuJoCo on the **same training model** differs
by at most 2.9614e-5 in next generalized velocity. This is a reset-state and
one-step comparison, not temporal-history, full-episode or hardware parity.

The stale replay gyro differs from current state on all 32 samples, by up to
1.4223 rad/s in a penetrating crawl sample. Corrected gyro maximum difference
is 2.3004e-8 rad/s. Large errors in invalid contact samples are not claims
about normal physical gyro noise or firmware behavior.

The pinned replay mesh inherits armature 0.01, damping 0.001 and friction loss
0.1 on all six **unactuated floating-base** coordinates; actual training has
zero on those coordinates. Actuated joint armature, damping/friction, body
mass/inertia and joint effort/position limits agree. Actuator-level limit
mechanisms and collision/contact settings are separately reported, not called
identical. Training/replay solver iteration budgets are 10/100 and line-search
budgets 20/50. These source models and defaults were not edited.

Root-only, solver/options-only and combined changes are tested on diagnostic
copies. Aligning only free-root properties removes free-flight next-velocity
differences on all 17 mutually contact-free samples (maximum residual
1.8214e-10). It does **not** remove contact-case differences; unmodified models
differ by up to 0.7699 in next generalized velocity. Neither increasing solver
iterations nor removing artificial base friction is established as a dance fix.

Authoritative evidence: `artifacts/g1_true23_frozen_lora/training_replay_parity_20260906_v3/`
contains `report.json`, `trace.npz` and the actual compiled training/replay
models. Report SHA256:
`d85815473d5a946664f0162811a90e6fb19e6300d4083228130c90f00c2fa44e`.
Inputs, policy pairing, output bytes, editable-runtime sources and package
versions are recorded. Development `_v1`/`_v2` reports lack parts of this
diagnostic/provenance coverage and are not the final implementation report.

### Full requested replays after the fix

Same model-50 pair, stance V2 references, all 23 joints, unchanged configured
gains, 35 Nm modeled ankle rating, quarter-effort/95% projection, 5 rad/s
target slew and 500/50 Hz cadence. No predictive filter or retraining.

| Case | Before timing fix | After timing fix | Current active physics steps |
|---|---:|---:|---:|
| Happy dance, reference start | 54/535 | 51/535 | 515 |
| Happy dance, recorded posture + 5 s standing | 76/535 | 75/535 | 754 |
| PICO upright reference | 50/1,013 | 46/1,013 | 465 |
| PICO standing reference | 44/1,013 | 39/1,013 | 394 |

All fail target-intersection feasibility. Reference-start failures now occur
at right hip pitch; measured-start failure remains left ankle pitch. Dance
maximum joint RMSE is 0.4175/0.4075 rad, above the provisional 0.35 threshold.
Recorded-posture acquisition still passes all 250 transitions/2,500 physics
steps, but the requested five-second return completes zero physics steps:
the ankle already requires 5.8499 rad/s instantaneous target slew versus 5.
This is the pinned standing compatibility actor, not a Unitree FSM handoff.

Evidence: `artifacts/g1_true23_frozen_lora/current_observation_timing_20260906_v1/`
(`dance_reference`, `dance_measured`, `pico_upright`, `pico_standing`). Fixing
observation accuracy slightly reduces survival counts; it does not improve
readiness. Previous counts are historical results with stale observations.
The remaining work is realizable full-body references, closed-loop native
balance/tracking training and verified acquisition/return—not bypassing guards.
No corrected matched-budget v14 comparison or arbitrary 29-to-23 motion parity
has been established. All readiness flags remain false.

Verification: **223 tests pass** (183 existing focused tests, 20 new parity/
timing regressions, nine PICO-builder tests and 11 legacy clean-replay tests).
Existing asset-dependent tests use current worktree code with the original
sibling's unchanged pinned assets; no model hash was relaxed. New audit/test
files pass full lint; changed existing modules pass lint with their pre-existing
long-line rule excluded. All 237 recorded audit input/source/output hashes
were independently rechecked after execution; all controller-status checks agree.

## 2026-09-06 continuation: full-body stance retargeting, still not live-ready

**Physical dance, standing return and live full-body teleop remain NOT ready.**
No robot connection/commands, mode changes, physical-controller or pin edits,
limit/interlock changes, training or policy promotion occurred in this pass.
Existing uncommitted hardware work remains untouched in the separate worktree.

### New offline implementation and verification

`gear_sonic/scripts/retarget_g1_true23_stance.py` now builds explicitly labeled
**candidate references**, not accepted teachers or executable robot bundles.
It optimizes all 23 joint coordinates plus floating-root XYZ, preserving root
orientation and all original 50 Hz samples. The eight-clip family file keeps
standing, walking, hand crawl, elbow crawl and original SONIC happy dance.
Contact schedules are hypotheses, not measured contacts; no upper-body fallback
or deletion of a difficult clip/frame is used.

The solver uses actual mesh and training-capsule geometry, flat-foot stance
targets, retained swing-foot/hand/elbow/torso objectives, COM support polygons,
and soft whole-body floor-distance penalties. Standing COM is centered over
the ankle origins, not merely nudged inside the toe edge. Analytic body,
orientation, COM and penetrating-floor Jacobians pass finite-difference tests.
The supplied models remain byte-identical at the compiled-model level.

Whole-path projection constrains joint corrections to 0.6 rad and root XYZ
corrections to 0.08 m per axis. Serialized joint paths are bounded by 5 rad/s
and 80 rad/s²; root **corrections** by 0.75 m/s and 6 m/s². These are offline
shaping bounds, not certified hardware ratings. Root correction bounds do not
bound the original root motion. Initial projection velocity is explicitly
derived from the candidate, not proof of acquisition from rest. Solver success
and the soft floor objective do not establish nonlinear contact constraints
after temporal projection; independent geometry/force audits remain decisive.

Final artifacts are in
`artifacts/g1_true23_frozen_lora/stance_retarget_20260906_v2/`: all eight NPZs,
per-clip reports, `motions.json`, `report.json`, `support_static/report.json`,
`support_dynamics/report.json`, `eval_reference/`, `eval_measured/`,
`eval_pico_upright_anchor/`, and `eval_pico_standing_anchor/`.
All **6,035 reference frames** remain; **5,955 complete q9/q10 history packets**
were rebuilt from the changed positions and schema/proof-validated offline.
Old packet hashes cannot be reused. No packets were sent or promoted. The
PICO anchors still contain the original short recordings plus terminal holds;
these are not newly recorded live teleop sessions.

Input/source/output hashes and both-model full-corpus audit counts were
independently rechecked after completion. Final report hashes:

- Retarget: `b3f77fa3b01dd5cb4763e17c3535b63f78f030d4ca97e5a66933d38f68e64d41`
- Static support: `6177185b9e76d028b3479901f65255e72a912aef5829207feae182f6d6588fa8`
- Reference dynamics: `c9dd9bcb9d9a8d4d0a5421704bdcd3ed7bdeef841af0b89d2bf5695baf9a61a8`

The development `_v1` variant lacked swing-foot/floor objectives and standing
COM centering; it is a rejected experiment, not the final implementation.
It placed standing COM near the toe edge and required approximately 16 Nm
ankle torque against the unchanged 8.3125 Nm modeled screen. A deterministic
saved-upright-like regression fixture distinguishes that failure from the
centered solution without changing effort bounds.

### Results: standing-reference improvement, no full-motion qualification

Upright and standing anchors now have **zero floor-overlap frames in both
models**. Each has a conditional in-limit force-balance solution on all
**1,024/1,024 frames**, in both the stationary and pose-derived inverse-dynamics
screens. Previously both had zero passing support frames after grounding alone.
This is a reference-level improvement, not proof of closed-loop standing,
contact complementarity, actual motor ratings, or physical teleop readiness.
The support audit retains its documented optimistic contact/friction assumptions.

The other six clips fail. Crouch still penetrates on six frames and lacks a
support wrench on most frames. Walking and crawling retain penetration and
support failures. Happy dance still has mesh/capsule overlap on **382/391 of
546 frames**, worst **37.59/41.03 mm**, with only **65/546** conditional in-limit
inverse-dynamics frames in either model. It changes a body position by up to
**0.5007 m** and reaches the correction bounds; this is not original-motion
parity. The bounded optimizer reports nonconvergence on 244 dance frames;
those diagnostics are retained rather than relabeled success. No candidate
corpus is accepted for production training.

### Unchanged correct-pair controller evaluation

The actual paired model-50 encoder `3806b2b6...` and decoder `eb3e0c06...`
were evaluated on all requested happy-dance transitions with the same V2
stateful native controller, configured-sim gains, full action fraction,
modeled 35 Nm ankle, unchanged quarter-effort/95% projection bounds,
5 rad/s target slew, and 500 Hz PD / 50 Hz policy. No predictive filter,
retraining or mismatched encoder was substituted.

| Reference variant | Reference-start transitions | Recorded posture + 5 s standing | Return qualified |
|---|---:|---:|---|
| Previous floor-conditioned reference | 50/535 | 16/535 | No |
| New whole-body stance candidate | 54/535 | 76/535 | No |

The changed reference and its fidelity defects prevent treating these counts
as physical progress or parity. New reference-start failure is right ankle
pitch at 549 active physics steps: required instantaneous target slew
11.5849 rad/s versus 5. Recorded-posture failure is left ankle pitch at 763
active physics steps: 5.7351 versus 5. Maximum joint tracking RMSE remains
0.4318/0.4277 rad, above the provisional 0.35 rad fidelity threshold.
Measured-start acquisition still passes its 250-transition/2,500-step standing
screen; the requested 5-second return completes **zero physics steps** because
the ankle target intersection is already empty. This uses historical posture
and a pinned 29-to-23 standing compatibility actor, not actual Unitree FSM
ownership/handoff or fresh robot telemetry.

The unchanged paired SONIC controller was also evaluated on the two improved
PICO references. Upright completes **50/1,013 transitions, 509 physics steps**;
standing completes **44/1,013, 444 physics steps**. Both fail left ankle target
feasibility (9.0338/9.0041 rad/s instantaneous need versus 5), with maximum
knee-flexion deltas 0.9932/0.8174 rad. Therefore even the newly supportable
standing references do not establish closed-loop teleop readiness. Reference
repair and policy/controller qualification are separate requirements.

Verification: **183 focused tests plus nine existing PICO-builder tests pass**,
including 16 new stance-retarget regressions; lint passes. Existing PICO tests
use the original sibling's unchanged pinned LF assets with current worktree
code. No tests bypass physical authorization or edit the model pin.

The physical bit-30 motors-off/damping cause is still unknown. This pass does
not establish recovery. Next work must enforce realizable contact/force and
motion-fidelity constraints jointly with temporal limits, then qualify native
closed-loop acquisition, full dance, standing return and live PICO input.
Missing axes prevent promising exact reproduction of every 29-DoF motion.
The comparison document distinguishes these reference experiments from a
matched-budget retraining comparison against original v14, which is still absent.

## 2026-09-06 continuation: reference force balance and PICO grounding

**Physical dance/live full-body teleop remain NOT ready. No robot commands,
mode changes, hardware-controller/pin edits, retraining or policy promotion.**
Whole-reference floor conditioning was committed and pushed as
`909311a13d81ed56987814ea1daea64d8c5ae5db`. This continuation identifies reference
support defects and adds an opt-in geometric correction, not a recovered robot.

### Force-balance audit, not a controller or qualification gate

New `audit_g1_true23_reference_support.py` checks all 6,035 frames of all eight
clips against both supplied mesh/capsule models. It solves for unilateral
ground-contact forces and all 23 joint torques, minimizing the largest torque
relative to the unchanged `0.95 * 0.25 * effort` bounds (8.3125 Nm at the modeled
35 Nm ankle). The six floating-base coordinates have no invented actuators.
It offers two explicitly separate hypotheses: stationary poses, or unsmoothed
50 Hz pose-derived velocity/acceleration and inverse dynamics.

The implementation follows [MuJoCo's force/Jacobian convention](https://mujoco.readthedocs.io/en/3.5.0/computation/index.html#contact).
Near-floor candidates use an explicit 2 mm gap tolerance and pyramidal cones;
they do not prove actual contact or contact complementarity. Bounded friction
assistance is optimistically selected without velocity-sign constraints.
Self-contact and joint-limit forces are omitted. Floating-base armature,
damping and omitted friction-loss parameters are reported, not silently fixed.
These are the supplied compiled reference models, not controller-matched
rollouts. Target-position/slew feasibility, dynamic tracking and hardware
readiness are **not** established by a force-balance solution.

Tests caught that changing runtime contact margins alone misses nearby toe
geoms through the compiled static BVH. The independent candidate copy now
disables collision midphase filtering; the original model is unchanged.
Degenerate HiGHS results marked Unknown are not treated as infeasibility:
the unactuated root-wrench LP can certify infeasibility, or the unchanged full
LP is retried without presolve. Remaining solver errors fail explicitly.
Returned solutions receive independent residual/unilateral/torque checks.

Successful complete reports are under
`artifacts/g1_true23_frozen_lora/reference_support_20260906_v3/`:
`original_static`, `original_dynamics`, `conditioned_static`,
`conditioned_dynamics`, `grounded_pico_static`, `grounded_pico_dynamics`.
Each contains `report.json`, inputs/source hashes, model hashes and per-frame
evidence. Earlier support `_v1`/`_v2` attempts stopped on solver diagnostics
and are incomplete, not authoritative corpus reports.

For happy dance, the **reference inverse-dynamics hypothesis** gives:

| Reference / model | No force-balance solution | Solution above effort bounds | Conditional solution within bounds |
|---|---:|---:|---:|
| Unconditioned / mesh | 179 | 278 | 89 |
| Unconditioned / capsules | 238 | 220 | 88 |
| Floor-conditioned / mesh | 349 | 122 | 75 |
| Floor-conditioned / capsules | 368 | 103 | 75 |

Every row covers all **546 reference frames**, not 535 policy transitions.
Unconditioned references contain floor penetration; their larger conditional
counts are not evidence of better physical feasibility. Floor conditioning
can remove support while clearing overlap. It is not sufficient retargeting.
Maximum finite minimum-effort ratios for conditioned dance are 6.6171 / 6.1838
in the two supplied models. This does not justify increasing physical limits,
prove the physical motor-off cause, or compare trained models against v14.

### Concrete PICO root-height defect and opt-in correction

The PICO builder's historical rule fixes the lowest **ankle body** at 0.06 m.
It ignores foot orientation/shape. In the stored upright/standing anchors,
the lowest mesh foot-floor gaps are 13.216–13.712 mm and 13.438–14.139 mm;
both clips have no candidate support on all 1,024 frames in either model.

New `--collision-grounding` uses actual foot collision geometry and a 10 μm
clearance. It preserves original defaults, all native23 q9/q10 joint proofs,
all joint position/velocity samples, body orientations/angular velocities,
horizontal paths and original timing. Only root/body vertical positions and
their existing forward-difference velocity convention change. It is specific
to this PICO builder's upright-root, stationary-XY, stance assumption; do not
apply it indiscriminately to jumping/dance clips. Other body-floor penetration
is rejected, not hidden by deleting frames. No derivative limits are claimed.

All three clips were rebuilt under
`artifacts/g1_true23_frozen_lora/pico_collision_grounding_20260906_v1/` with
separate NPZ/report files. The eight-clip `motions.json` retains the other five
floor-conditioned clips; no clip is removed and no production corpus changes.
The anchors retain **51 / 48 / 72 source frames**, extended by the original
**973 / 976 / 952 terminal-hold frames**. They are not 20 seconds of newly
recorded live PICO motion. Original files and packet hashes are unchanged.

The unchanged byte-pinned LF model/config in the original sibling repository
was read for these builds; outputs and edited code remain in the separate
worktree. Its CRLF checkout of the same XML fails the existing byte pin (the
LF-normalized hash matches); no pin or model file was relaxed/edited.

The correction fixes mesh hover but **does not fix support**. All three PICO
clips still have zero conditional in-limit frames in both audit modes/models.
Grounded upright in the capsule model has a solution but exceeds the limits
on every frame (up to 2.2771 times, left ankle pitch). Mesh upright/standing
and crouch lack a compatible support wrench. At the first standing frame the
mesh candidates are on the left foot only, while COM lies between the feet.
The capsule model also detects floor overlap on 1,024 / 1,024 / 970 frames of
these mesh-grounded clips. Therefore **do not train/promote these candidates**
as support-feasible teachers. Next retargeting must jointly correct foot
orientation/placement, lower-body posture, COM and force feasibility across
the chosen dynamics/contact model, rather than change root height alone.

Verification: **167 focused tests pass**, plus **9 existing PICO-builder
regressions** using the unchanged original asset root (176 total, no skipped
tests). New tests cover analytic support torques, push-only contacts, no fake
base actuation, MuJoCo wrench mapping, candidate-gap enumeration, solver
Unknown handling, quaternion derivatives, ideal freefall versus floating
standing, whole-clip preservation, opt-in behavior and explicit false
readiness. All six owned Python files pass critical/import Ruff and formatting.
Separate NPZ hash/array checks confirm the three rebuilt clips preserve the
named source channels exactly. Existing dirty hardware paths remain excluded.
All six complete force reports also pass post-run input/source-hash and frame
count checks. Compared with the historical crouch NPZ, regenerated horizontal
linear velocities differ by at most 1.39e-15 m/s in 26 entries; the same
roundoff occurs with today's unchanged legacy builder. Grounded versus current
legacy horizontal velocities are bit-identical. This is not a grounding-induced
horizontal-motion change.

Full-dance controller results remain the prior **50/535 reference-start and
16/535 historical measured-start**, with failed immediate standing return.
No new matched-budget v14 comparison, physical motor-off diagnosis, Unitree
handoff qualification, encoder-pin repair or calibrated live PICO run occurred.

## 2026-09-06 continuation: whole-reference floor conditioning

**Physical full-body dance/live teleop remain NOT ready. No robot commands,
mode changes, controller/pin edits, retraining or model promotions.** Previous
exploration/reset work was committed/pushed as
`61769ea5fe82016024cb188105acde0275376a42`. This continuation changes separate
offline reference artifacts, not the deployed motion or controller.

### Contact geometry and complete-path correction

Retargeting/evaluation use `g1_23dof_rev_1_0.xml` with mesh body collisions and
four foot spheres per side. Training uses Unitree MJLab's `g1_23dof.xml` with
capsule body collisions and seven foot capsules per side. Same 23 joint names
do not imply identical contact geometry. Both models already intersect the
floor on many source frames, so model mismatch alone is not the explanation.

The new CPU training-geometry builder uses the actual training `Entity` and
collision configuration. Against all 32 initial qpos snapshots from the prior
actual compiled MJLab rollout, minimum floor distances match **exactly** and
all floor-penetration contact counts match. This validates that geometric
comparison, not dynamics/solver equivalence or a hardware shape model.

New `condition_g1_true23_reference_floor.py` audits every frame, with optional
`--condition`. It creates a separate, whole-horizon minimum-norm root-z
correction clearing **both** supplied collision models, bounded by 0.2 m lift,
0.75 m/s correction velocity and 6 m/s² correction acceleration. These are
offline correction bounds, not verified hardware limits or bounds on the
original root trajectory. Correction starts with zero offset velocity and
targets 10 micrometres geometric clearance. Bounds are independently checked
after float32 serialization, followed by fresh contact checks.

All eight source clips and **6,035/6,035 frames** remain at original 50 Hz.
Joint positions, joint velocities, every body quaternion/angular velocity and
FPS are preserved bit-for-bit. Horizontal body positions/velocities are also
unchanged. Only body z positions and the correction's vertical-velocity term
change. Input body arrays must first agree with native23 FK; inconsistent
channels are rejected, not silently rebuilt. Original files remain untouched.

| Full clip | Frames | Overlap frames: mesh / training | Maximum root lift (m) |
|---|---:|---:|---:|
| PICO upright anchor | 1024 | 0 / 0 | 0 |
| PICO standing anchor | 1024 | 0 / 0 | 0 |
| PICO crouch anchor | 1024 | 957 / 957 | 0.008874 |
| PICO walk 001 | 695 | 49 / 63 | 0.022843 |
| PICO walk 010 | 510 | 32 / 43 | 0.030472 |
| Original SONIC hand crawl | 606 | 549 / 555 | 0.117508 |
| Original SONIC elbow crawl | 606 | 600 / 604 | 0.028057 |
| Original SONIC happy dance | 546 | 522 / 512 | 0.015906 |

After conditioning, **zero frames have detected floor overlap in either
model**. This proves only the named geometric property. Root lifting does not
optimize support-contact placement, COM, contact forces, actuator torques,
self-collisions or dynamic feasibility. Raising a pose can also reduce ground
support; these files are not accepted teachers or ready-to-run robot motions.

Authoritative artifacts:
`artifacts/g1_true23_frozen_lora/reference_floor_20260906_v2/`, containing all
eight NPZ files, `motions.json` and `report.json`. The report binds source
files/inputs and complete compiled-model MJB hashes. The earlier `_v1` is
preserved as a development variant: it re-estimated joint/angular velocity
channels during FK export and was **not** used for the full-dance tests below.
V2 preserves those source channels exactly.

### Full-dance and measured-start checks still fail

The unchanged model50 and actual paired encoder were evaluated with V2
references, unchanged native-support V2 gains, 35 Nm modeled ankle limit,
quarter-effort guard, 5 rad/s slew and applied-target feedback:

- Reference start: **50/535** completed transitions, **505** active physics
  substeps, then a right-ankle empty target intersection. Maximum joint RMSE
  0.42285 rad; full-clip fidelity fails. No standing return was requested in
  this reference-only case.
- Historical measured state + 5 s standing: standing startup passes the
  existing simulator screen; dance still stops at **16/535**, **164** physics
  substeps. Immediate return completes **zero physics steps** before another
  empty intersection. This does not qualify a recovered or standing robot.

Evidence: `eval_reference/summary.json` and `eval_measured/summary.json` within
the V2 directory, with 500 Hz traces. The full-motion completion counts are
unchanged from unconditioned references (50 and 16); floor conditioning is not
the missing control fix. These are corrected-reference fidelity screens, not
proof of original-v14 parity. The measured snapshot remains historical, and
the standing compatibility actor is not Unitree FSM ownership/handback.

No new training was started merely because geometric checks pass. Next work
must resolve support/contact-force and actuator feasibility, then alter
native23 task-space references/learned actions where necessary. Current
retargeting, gains and safety-constrained controller still do not establish
stable full-body tracking. Physical motor-off cause, encoder pins, Unitree
handoff and calibrated PICO input remain unresolved prerequisites.

Verification: **140 focused tests pass** across the deployment envelope,
simulation acquisition, applied-target feedback, predictive filter, retiming,
stage-one actuation, reset feasibility, training integration, exploration audit
and new whole-reference geometry tests. Critical/import Ruff checks and format
checks pass for the five owned Python files; owned-path `git diff --check`
passes. Existing dirty hardware launchers/controller paths are excluded from
this offline milestone and are not overwritten or staged.

## 2026-09-06 continuation: exploration and reset-contact diagnosis

**Physical full-body dance and live teleop remain NOT ready. No robot commands,
mode changes, physical-controller edits, pin changes or model promotions.**
The previous offline feasibility work is committed/pushed as
`71eb438031007a0786da3ead6cf7277935fc0d75`. This continuation investigates why
constrained training episodes end after only a few control steps; it does not
claim to explain the physical motor-off latch or repair normal-mode handoff.

New `gear_sonic/scripts/audit_g1_true23_exploration_rollout.py` runs the actual
MJLab training environment and frozen model50 without a runner, optimizer or
weight updates. It preserves the eight-clip SONIC/PICO corpus, all 23 controlled
joints, V2 target feedback, the 2 ms PD / 20 ms policy periods, and existing
hard-margin, effort and 5 rad/s target-slew limits. It checks frozen weights,
input/source hashes and the union of actual termination masks against wrapper
done signals. Unfinished episodes stay censored, not successful full clips.

### Exploration noise is not the sole cause

The original v14 trainer fixes Gaussian action std at **0.10**. Frozen LoRA
inherits the released per-joint std (native23 mean **0.3845523**). Three
no-learning probes scaled only action exploration, using seed 20260906,
32 environments and 128 control steps each (4,096 sampled transitions/case):

| Action-noise scale | Completed episodes | Mean / median completed steps | One-step episodes | Actuation-guard terminations |
|---|---:|---:|---:|---:|
| 1.0 | 935 | 4.124 / 3 | 142 | 921 |
| 0.25 | 922 | 4.106 / 3 | 127 | 911 |
| 0.0 | 847 | 4.433 / 3 | 126 | 834 |

All three share input/source hashes and the logged initial joint/controller/
observation/phase hash `fef5718157ee0138005605e682d92eaee711072f802ba1330838770ed7ef5096`.
The first batch's 32 episodes all end within the observation window; their
mean lengths are 6.3125, 4.625 and 4.1875 steps respectively. Later adaptive
reset sequences diverge and are not paired. Observation corruption remains
enabled for tokenizer and policy even at zero **action** exploration. This is
not a noise-free sensor test, a multi-seed quality estimate, a v14-weight
comparison, or evidence supporting a production std change. No retraining
occurred and the rejected model50/paired encoder are unchanged.

### First substep failures and floor overlap

Optional failure capture independently checks Torch-flagged states using the
float64 NumPy target intersection. All first 128 captured failures in the
32-step `mean_failure_trace` probe confirm empty intersections. Of 130
offending-joint entries, 104 are ankle pitch. None of these intervals is empty
when previous-target slew is removed; this does not authorize a target jump.

Two initial crouch states have feasible controller seeds and zero ankle
velocity, yet fail at physics substeps 4 and 7. Their left ankle velocities
reach -4.318 and -3.574 rad/s; instantaneous required target slew is 16.339 and
8.949 rad/s against the unchanged 5 rad/s limit. A separate CPU positional
forward pass on the actual compiled model finds **15/32** initial poses with
floor overlap, worst **0.106392 m**. Those two crouch states overlap the floor
by **0.051014 m** and **0.029234 m**. Another first-control-step failure starts
with no detected floor contact, so reset penetration cannot explain every
failure by itself.

The contact probe uses independent MuJoCo data and performs no integration;
negative distance is geometric overlap, not proof of a particular impulse.
See [MuJoCo's pipeline API](https://mujoco.readthedocs.io/en/stable/APIreference/APIfunctions.html#mj-fwdposition)
and [contact model](https://mujoco.readthedocs.io/en/stable/computation/index.html#contact).
Random resets perturb root position/orientation, root linear/angular velocity
and joints (joint position ±0.1 rad); disabling event randomizers did not
disable these motion-command perturbations.

Repeated zero-action-noise runs are not bitwise identical beyond initialization:
the first observed q difference is 8.05e-6 rad at control step 1, with done
sequences first diverging at step 17 in the checked 32-step prefix. The source
of this small numerical variation has not been isolated. The observer returns
the original PD result unchanged, but trajectory identity is not claimed.

Evidence is under
`artifacts/g1_true23_frozen_lora/exploration_audit_20260906_v1/`, with
`{full,quarter,mean}/summary.json`, `mean_failure_trace/summary.json` and
`mean_reset_contacts/summary.json` plus their hash-bound rollout arrays.
Two attempted parallel source-load processes failed with host-RAM OSError 12
before producing rollout evidence; serial reruns succeeded. This was not a
robot or model failure. Runs loading the large frozen source should remain
serial on this 8 GB WSL host.

### Reset interventions isolate a contributor, not a working policy

The audit has two explicit, simulator-only switches:

- `--reset-perturbation-scale 0` removes command-reset perturbations while
  retaining source poses/velocities, soft-joint clipping and observation noise.
- `--lift-reset-floor-overlap` minimally raises only penetrated reset states,
  with 10 micrometres clearance and a 0.2 m rejection bound. It supports one
  floating articulation and one horizontal world-welded plane; it rejects
  movable/tilted floors and unrelated geometry. Airborne states are not
  lowered. Every applied reset asserts unchanged velocities and exact expected
  qpos. No physics settling, reference modification or joint removal occurs.

Four same-source/input-hash runs use zero action exploration, the same
initial sampled phases, seed 20260906 and 4,096 transitions each:

| Reset case | Completed episodes | Mean / median completed steps | One-step episodes | Guard terminations |
|---|---:|---:|---:|---:|
| Original randomized resets (`reset_baseline128_v2`) | 837 | 4.612 / 3 | 128 | 822 |
| Randomized resets + floor lift (`reset_lift128`) | 1000 | 4.041 / 3 | 47 | 998 |
| Unperturbed reference resets (`reset_nominal128`) | 828 | 4.564 / 3 | 18 | 821 |
| Unperturbed resets + floor lift (`reset_nominal_lift128`) | 898 | 4.355 / 3 | 22 | 885 |

The floor-lift case corrects 525 of 1,032 reset rows (maximum lift 0.156665 m),
leaving zero detected floor penetration at those reset checks. Initial joint
q/dq, controller targets and full generalized velocities match baseline
exactly; only root z differs in qpos. First-policy-interval failures drop from
4/32 to 1/32. The two crouch failures move from physics substeps 4/7 to 22/29,
but still fail. The first batch's mean episode length decreases from 4.1875 to
3.78125 steps. Overall median remains three. This supports a reset-contact
contribution, not a quality gain, complete causal explanation or useful policy.

Removing reset perturbations also fails to eliminate overlap: **17/32** nominal
states intersect the compiled floor, worst **0.101913 m**. The native23
reference/contact geometry therefore needs review as well as randomization;
neither perturbation removal nor root lifting is promoted into training.
Combining both corrections also fails: it lifts 424/930 reset rows (maximum
0.103775 m), leaving zero detected reset overlaps, but still has 885 guard
terminations and three-step median completed episodes. Its first batch's
mean is 5.09375 steps, median three, with one first-interval failure. This
limited mean increase is not full-motion tracking or demonstrated improvement.
After first termination, comparisons are unpaired adaptive rollouts. Earlier
`reset_baseline128` (854 guards) is retained but predates the welded-floor
helper/assertions; the table uses the matched-source `reset_baseline128_v2`.
Repeated baseline totals vary and are not a statistical improvement measure.

Artifact identity clarification: these no-learning runs load the adapter/
optimizer checkpoint `breadth50/checkpoints/frozen_lora_model_50.pt`, file SHA
`9e916b9cdfcfa60890bdd1893aa3faabdaf9dc354fd600190aba533bbe494a83`, with lineage
`c65e698a28b48cdd3414ceca5314bc16b69a516a40d1cbdf921a224252173548`.
The older `848f2bd6...` SHA identifies **`eval/model_50.diagnostic.pt`**, the
materialized policy file used by previous ONNX exports, not this resumable
checkpoint. No artifact changed; the earlier shorthand label is clarified.

Verification: **129 focused tests pass**, including 19 new audit tests and the
110 existing feasibility/controller/trainer tests. Formatting, import/critical
Ruff and diff checks pass. All ten saved rollout reports were independently
recounted from their NPZ termination arrays and checked against recorded file
hashes, zero-optimizer-step flags and false deployment/hardware flags. The four
final reset cases have identical source/input hash sets; initial physics hashes
explicitly include root qpos/qvel and differ as expected across interventions.
These tests validate software/evidence behavior, not robot readiness.

Next: review native23 reference collision/contact geometry and generate
contact/COM-consistent, multi-step feasible training references before another
bounded learning run. Any candidate still needs full-clip fidelity and actual
measured-start standing/dance/return evaluation under a hardware-reviewed
profile. The motor-off latch, deployed encoder pairing, normal-mode handoff
and calibrated PICO tracking remain separate unresolved physical prerequisites.
Neither more blind updates nor a physical retry is justified by these probes.
All six missing axes remain absent; arbitrary 29-DoF motions require individual
retargeting and qualification, not a guarantee of exact 29-DoF reproduction.

## 2026-09-06 continuation: predictive feasibility and full-path retiming

**Physical full-body dance and live teleop remain NOT ready. No robot commands,
mode changes, hardware-controller edits, pin changes or model promotions.**
This continuation adds offline diagnostics and rejected experiments, not a
physical damping fix. Previous controller-state correction was committed and
pushed as `e95cb7e137bdf01324dfdf1a945b14db87fdc517`.

### Joint-level failure evidence and copied-simulator prediction

The existing effort projection now reports the exact joint, state, previous
target and empty interval without changing its rejection boundary. Optional
500 Hz traces preserve q/dq, requested/applied targets, effort and acceleration;
terminal target and actual partial physics steps remain separate from completed
50 Hz control intervals.

- Original-tempo reference start fails at the right ankle pitch after 504
  physics steps (50/535 completed control transitions). Its effort-feasible
  interval requires at least **13.8856 rad/s instantaneous target slew** from
  the previous target; configured slew remains **5 rad/s**.
- Historical measured start plus 5 s standing fails at the same joint after
  164 physics steps (16/535 transitions), requiring at least **7.1268 rad/s**
  at that instant. Both states admit a target if slew memory is omitted, but
  that does not authorize a discontinuity or establish dynamic recoverability.

Intrinsically empty effort/position intervals instead report no finite slew
solution (`null`); increasing slew cannot make those intervals intersect.

New opt-in `--predictive-active-effort` uses an independent copied MuJoCo state
to test next-step target feasibility before executing the current target.
Bounded sequential linearization keeps all 23 targets free within unchanged
hard-margin, quarter-effort and 5 rad/s slew bounds. Every accepted target is
checked with the nonlinear preview. Tests verify that previews do not mutate
the actual simulator and match its next physics step. Search failure means
**this bounded optimizer found no verified target**, not global infeasibility.
This filter is neither part of the trained policy nor a hardware controller;
one-step existence does not prove multi-step feasibility or safe return.

The experiment ran under MuJoCo 3.5.0 / SciPy 1.16.2. The preview also uses
`MjData.__copy__` when `mj_copyData` is unavailable, matching the copy API in
the repository's pinned MuJoCo 3.2.3 ([upstream binding](https://raw.githubusercontent.com/google-deepmind/mujoco/3.2.3/python/mujoco/functions.cc)).
Both copy paths are tested with actual MuJoCo data in the installed 3.5.0
runtime; that is not a complete 3.2.3 training/runtime qualification. No
dependency pin was changed.

All runs use the previously rejected V2 model50, with its actual paired
encoder; **no additional training occurred** in this continuation:

- Materialized diagnostic policy file SHA: `848f2bd69847198594278373c5e1f96557bbaf7b39b6947ef6088ed3393af3f8`.
- Decoder SHA: `eb3e0c06836d3be88c27ace59e428dbf9826ffb449f65d80b5b9317618a19796`.
- Encoder SHA: `3806b2b63ebadf4d6cbf9f79b7072f2bf27ab8eb8bc6a9b3042f97739cc5428a`.

Original-tempo predictive reference run still completes only **50/535**,
stopping at 503 physics steps before the greedy run's next conflict. Measured
start improves from **16/535 to 72/535**, with 722 physics steps and 36 successful
interventions, but full-clip fidelity still fails (maximum joint RMSE 0.3801
rad). Its attempted return completes only **one 2 ms physics substep**, not a
complete 50 Hz interval, before a left-ankle empty intersection. Standing
startup passing numeric checks is not a successful dance or recovery.

Successful original-tempo filter calls took up to **16.106 ms**; the final
failed searches took 95.073 ms (reference) and 46.895 ms (measured). The report's
`preview_calls` and `maximum_filter_elapsed_s` count only successful filter
returns; failed-search timing and preview count are separately in
`failure.details`. This implementation is not qualified for the 2 ms loop.

### Half-speed experiment preserves the full reference, still fails

New `retime_g1_true23_sonic_reference.py` produces a separate standard motion
NPZ and hash-bound audit sidecar. At 2x duration, all **546 source joint samples
are preserved exactly** across 1,091 frames / 21.8 s, with native23 FK and
velocities recomputed. No controlled joint, phase interval or root-path segment
is removed. Contact/COM optimization and dynamic feasibility are not claimed.
The first control frame corresponds to source phase 5 rather than original
phase 10, so this is not a same-initial-state, timing-only ablation. Slower
tempo and the longer denominator also preclude original-tempo parity claims.

| Half-speed start / filter | Completed transitions | Active physics steps | Result |
|---|---:|---:|---|
| Reference / greedy | 45/1080 | 450 | Right-hip target intersection empty |
| Reference / predictive | 49/1080 | 490 | Bounded next-step search fails, left ankle |
| Measured + standing / greedy | 82/1080 | 826 | Right-hip target intersection empty |
| Measured + standing / predictive | 86/1080 | 860 | Bounded next-step search fails, left ankle |

All four full-clip fidelity screens fail. The measured greedy return completes
zero physics steps; the measured predictive return completes one before
failure. Reference-only cases requested no standing return and therefore
cannot qualify a lifecycle. Slowing the clip did not solve the problem.

Evidence, relative to `artifacts/g1_true23_frozen_lora/`:

- `actuation_trace_20260905_v1/{reference,measured}/summary.json`.
- `predictive_projection_20260905_v1/{reference,measured}/summary.json`.
- `retiming_feasibility_20260906_v1/{reference,reference_predictive,measured,measured_predictive}/summary.json`.
- Valid retimed input: `retiming_feasibility_20260906_v1/happy_dance.slow2.standard.npz`,
  SHA `dbcd628ad7c9d4acdcbab75e55bf9a05f22938da743fbd14f65cf4cfbb89ce70`.
  The earlier `happy_dance.slow2.npz` has an unsupported extra channel, is
  preserved as failed format evidence, and was **not** evaluated. The standard
  file stores source-phase mapping in its JSON sidecar, not an extra NPZ key.

Verification: **110 focused tests pass** across deployment-envelope,
acquisition, controller state, predictive filtering, retiming, actuation,
reset-feasibility and trainer-contract tests. Formatting, import/critical Ruff
and diff checks pass. This does not erase known older artifact/pin test failures
or qualify the physical controller. Existing dirty hardware changes are
excluded from this continuation's commit.

The original-v14 comparison now labels its older loose-envelope, wrong-encoder
completion results as historical rather than fidelity parity. No new
matched-budget v14 comparison under corrected pairing and these constraints
has been completed. Physical encoder pins, motor-off cause, normal-mode
handoff and live PICO tracking/calibration remain unresolved; old motor-health
samples are not current readiness evidence.

Next work must improve policy tracking and multi-step feasible target/reference
generation before repeating training and measured-start lifecycle tests.
Neither slower playback, more blind updates, a late mode-switch patch nor
relaxed physical guards is supported as the fix. Missing six axes also mean
29-DoF motions require per-motion retargeting and qualification, not a promise
that every original pose is exactly reproducible on 23 DoF.

## 2026-09-05 continuation: consistent controller state, candidate still fails

**Physical full-body dance and live teleop remain NOT ready. No robot commands,
mode changes, physical-controller edits, gain-pin changes or promotions.**
The preceding projected-actuation work and rejected model were committed and
pushed separately as `fe4af55244cf6c9d03a1a50320d0d345acbb4560`.

### Simulator-only V2 reset and feedback correction

- New opt-in `--actuation-profile native_support_stateful_v2` preserves the
  existing profiles. Its distinct contract initializes a synthetic training
  controller target from the **final** reset q/dq, after the motion command
  reset, before previous-action observations. The target minimizes initial
  absolute PD effort within the unchanged hard-margin/effort interval.
- Every subsequent physics step retains the existing 5 rad/s slew and
  quarter-effort guard. Impossible seed rows stay marked for termination;
  processing a new action cannot erase that failure. Other environments'
  controller states are not reset. This is a synthetic initial-condition
  distribution, **not a physically reachable history or a hardware command**.
- V2 previous-action history encodes the last applied target, after slew and
  projection, in native23 action units. It does not report the unexecuted
  requested target or apply the tanh transform a second time. Initial history
  uses the observation manager's repeated synthetic reset state; it is not
  labeled a measured ten-frame physical history.
- `--stateful-native-controller` in the deployment-envelope diagnostic matches
  these feedback/reset semantics. Reference initialization is explicitly
  synthetic. Measured/neutral starts require a preceding balance controller;
  its actual terminal target is carried into SONIC, never reseeded afterward.
- Native-support behavior banks must bind the full controller-state contract,
  not only the simulator JSON hash shared by V1 and V2. Old bank claims cannot
  silently qualify changed feedback semantics.

Float32 nominal happy-dance audit: **3/532 reset seeds remain infeasible**;
all other seeds admit the first projected substep, with no additional failures.
This removes the q-seed artifact (206/532 in V1), not the three intrinsically
infeasible nominal states, randomized-reset risk, or dynamic-balance problem.

### Fresh training and full-clip checks

New run, no resume or artifact overwrite:
`artifacts/g1_true23_frozen_lora/native_support_stateful_20260905_v1/breadth50/`.
Exactly **50 updates / 25,600 transitions**, 32 environments, 16 rollout steps,
seed 20260905. Training loop took 129 s, excluding setup. No training process
remains running. Final mean episode length is **5.21 steps** versus V1's 3.90;
this single final-window statistic is not a controlled quality improvement.
Actuation guards still dominate terminations. Initial 32-environment prime
required no reset retries and reported no initial actuation violations.

All **31 source hashes** matched after training, including four files under
the actual `external_dependencies/unitree_rl_mjlab` source root. Lineage SHA:
`c65e698a28b48cdd3414ceca5314bc16b69a516a40d1cbdf921a224252173548`.
Diagnostic checkpoint SHA:
`848f2bd69847198594278373c5e1f96557bbaf7b39b6947ef6088ed3393af3f8`.
Decoder SHA:
`eb3e0c06836d3be88c27ace59e428dbf9826ffb449f65d80b5b9317618a19796`.
Decoder export passes three-case parity (max absolute error 1.550e-6);
paired encoder remains `3806b2b6...`, with exact three-case token parity.

- V2 synthetic reference start: **50/535 transitions**, 504 physics substeps,
  then empty effort/position/slew intersection. Max joint RMSE 0.4233 rad;
  max pelvis-position error 0.1537 m. Full-clip fidelity fails.
- Historical measured start + 5 s standing: startup passes all existing
  numeric guards; V2 SONIC fails at **16/535**, 164 physics substeps. Immediate
  return is infeasible at height **0.7742 m**, tilt **0.0212 rad**. Full dance
  and lifecycle both fail. This is not a successfully recovered robot.

Evidence: `eval/model_50.diagnostic.{encoder,decoder}.json`,
`eval/screen_reference/summary.json`, `eval/screen_measured/summary.json` within
the run directory. V2 changes controller feedback and reference-reset history,
so its numbers versus V1 are not a weights-only ablation. No model selected,
no original-v14 parity claim, no physical readiness claim.

Both saved terminal q/dq states admit some target within hard/effort bounds
when the previous-target slew constraint is omitted. This narrows the observed
failure to the intersection including controller target memory; it does not
authorize a target discontinuity. One-step greedy effort projection is not a
forward-reachability or recovery guarantee. Next work must examine early
target/velocity feasibility and learn or filter actions before entering these
states, with measured acquisition preserved. More blind updates, relaxed
limits, or a physical retry are not justified by this candidate.

Verification: **94 focused tests pass**, plus critical Ruff and diff checks.
Coverage includes reset-order/partial-reset tests, latched invalid seeds,
Torch/NumPy equivalence, old-bank rejection and an actual two-step MuJoCo test
showing decoder history receives the executed target rather than zero requested
actions. Source changes remain simulator-only; existing dirty hardware work is
not staged with this continuation.

## 2026-09-05 continuation: projected native actuation and reset feasibility

**Physical dance and live teleop remain NOT ready. No robot commands, mode
changes, hardware pin edits or promotions in this continuation.** The following
is a simulator-only training hypothesis, not a reviewed physical gain profile.

### Implemented and trained, with existing numeric limits

- New opt-in `--actuation-profile native_support_projected` uses the native
  simulator-configured gains, full safe SONIC targets, 5 rad/s target slew,
  explicit 500 Hz PD / 50 Hz policy, and a hypothetical 35 Nm ankle model cap.
  Target projection intersects hard-joint margins, target slew, and 95% of
  the existing quarter-effort guard. No limit was relaxed. Empty intersections
  latch training termination; zero effort until the next training reset is
  explicitly **not** a robot recovery behavior.
- Deployment-envelope diagnostics support matching `--project-active-effort`.
  Failures preserve actual partial-substep count, elapsed time and terminal
  qpos/qvel before an attempted return. They cannot become false completed
  control intervals. Raw action magnitude equal to 10 now also terminates
  profiled training, matching the runtime's rejection boundary.
- Actual learning: **50 updates, 25,600 transitions**, 32 environments and 16
  rollout steps. Directory is `native_support_training_20260905_v1/breadth200`,
  but **200 updates were only planned, not executed**. Final mean episode
  length was 3.90 steps. No training process remains running. All 30 source
  hashes in lineage matched the working tree at the end of this batch;
  lineage SHA is `0be71033b1567fc0a6a6a06c185498f70f2d4faa40ca303e0a4d9ae58da8f962`.

The diagnostic model50 checkpoint SHA is
`bfa5d7ce7ecafde54cc0e076d827479cace0adf7009e973bb951f1680ed0c0b5`.
Its decoder SHA is `f475e84d29f1f23cca98ac5013506a321d2e14c05526674175982bf951e34c7e`;
three-case Torch/ONNX parity passes (max absolute error 1.222e-6). The actual
paired frozen encoder remains `3806b2b6...` below, with exact three-case token
parity. Neither this export nor source matching establishes deployment safety.

### Closed-loop candidate rejected

All cases below use configured native gains, 35 Nm ankle cap, 5 rad/s slew,
full targets, active effort projection, and the actual paired encoder.

| Candidate/start | Completed dance transitions | Actual active physics steps | Result |
|---|---:|---:|---|
| Original breadth25 / reference | 86/535 | 864 | Empty target intersection; incomplete/fidelity fail |
| Newly trained native-support model50 / reference | 62/535 | 629 | Empty target intersection; joint RMSE max 0.4002 rad |
| New model50 / historical measured state + 5 s standing | 65/535 | 652 | Startup passes; dance fails; immediate return infeasible |

New model50 did not improve completion. Its failed return begins at height
0.7406 m and tilt 0.0413 rad: an empty intersection can occur while roughly
upright, not only after a fall. No full dance, lifecycle, original-v14 parity
or physical-readiness pass is claimed. Residual +0.25 with active projection
also fails (70/535 in the configured 5 rad/s case).

Evidence, relative to `artifacts/g1_true23_frozen_lora/`:

- `active_projection_20260905_v1/` preserves baseline/residual experiments.
- `native_support_training_20260905_v1/breadth200/eval/` contains exports and
  `screen_reference/summary.json`, `screen_measured/summary.json`.
- `native_support_training_20260905_v1/reset_feasibility_final.json` contains
  the reset audit below. Earlier evidence is retained, not overwritten.

### New reset-state defect; next work

Training currently seeds previous actuator target from joint position even
when a sampled motion reset has nonzero velocity. Under this profile, **206
of 532 eligible nominal happy-dance reset frames** then have no feasible
first target, regardless of the policy action. Only **3/532** remain
infeasible when considering any previous target inside the hard/effort
interval. This audit uses soft-clipped reference q and recorded dq before
random reset perturbations. It is not an audit of actual randomized resets,
reachable controller histories, or dynamic balance.

Next: model a consistent reset/controller state and its previous-action
history, distinguish random-reference resets from measured cold acquisition,
and re-evaluate early recovery feasibility. Do not simply loosen guards,
fabricate a physical history, or continue the remaining 150 training updates
under changed source while claiming exact resume. The old physical encoder
pin, motor-off cause and live PICO calibration remain unresolved.

Verification: **84 focused tests pass**, including Torch/NumPy projection
equivalence, isolated infeasible batch rows, raw-action boundary rejection,
partial-step reporting, CLI profile binding and reset-feasibility cases.

## 2026-09-05 continuation: wrong encoder pairing confirmed

**Physical dance and live teleop remain NOT ready. No robot commands, mode
changes, gain changes, hardware pin changes or promotions in this continuation.**
The old physical launcher still pins the mismatched encoder described below;
these offline fixes do not qualify that launcher or its controller.

### New evidence: selected encoder was not the training encoder

The original selected frozen-LoRA breadth25 checkpoint and the new breadth50
checkpoint use the same frozen encoder parameters:
`3625edb10aabd266196702aefd464ad07c93847f2d1722a977e18ef2a0143990`
(runner tensor-state fingerprint). Their freshly exported encoder ONNX files
are also byte-identical:
`3806b2b63ebadf4d6cbf9f79b7072f2bf27ab8eb8bc6a9b3042f97739cc5428a`.

The previously selected causal-model-250 encoder is instead
`733353148bef1eb8dd83a96416b7a89f0b5c3530ceb9e0cec9c25fdb04f56ff2`.
Direct audit against the original breadth25 training encoder disagrees on
**535/535 happy-dance causal reference inputs**, with maximum token-coordinate
error **0.9375**. Earlier breadth50 audit disagreed on 33–64 coordinates out
of 64 per frame. This is a real model-pairing fault, not an ABI-size mismatch
or harmless floating-point error. Matching encoder audit passes all 535
inputs, with exact token equality. Scope is this reference-input distribution,
not a live headset or all off-reference states.

Evidence (relative to `artifacts/g1_true23_frozen_lora/`):

- `balance_lifecycle_20260905_v1/encoder_parity.json`
- `balance_lifecycle_20260905_v1/breadth25_matching_encoder_parity.json`
- `paired_encoder_20260905_v2/original25_legacy_encoder_audit.json`
- `paired_encoder_20260905_v2/original_breadth25/model_25.diagnostic.{encoder,decoder}.json`

The newly exported original decoder has SHA `f4416889023eb629656fa189649d8cd071cdc3ae61fc1bfd888d07815d21bdc8`,
not the historical `c12038...` ONNX. Its initializer names and every tensor
are identical; both have 25 nodes. Exporter versions differ (Torch 2.10 vs
2.9). It passes fresh Torch/ONNX parity and reproduces the earlier corrected-
encoder baseline below. No old artifact was overwritten.

### Implemented: pairing and diagnostic correctness

- Diagnostic encoder export reconstructs the actual frozen encoder plus FSQ
  and requires exact discrete-token parity. Both export reports include the
  encoder parameter fingerprint and the static 267→64 / FSQ32 contract.
- `g1_true23_diagnostic_pair.py` validates both model files, report hashes,
  checkpoint identity, encoder identity, ABI, parity evidence and diagnostic-
  only flags. Missing/mixed legacy sidecars require re-export, not a pin edit.
  This is local provenance checking, not signed attestation or fresh physics
  qualification. The normal ONNX loader still checks actual runtime ABI.
- Happy residual fitting now requires `--encoder-report` from the same
  checkpoint as `--base-decoder-report`; its manifest binds the exact fitting
  encoder and base pair. No implicit causal-model-250 encoder for new fits.
- Deployment-envelope evaluation requires paired reports or a validated
  `--residual-manifest`. Explicit `--allow-unpaired-diagnostic` remains only
  for historical/mismatched-pair experiments; their paired-lifecycle screen
  can never pass. The old survival-only residual evaluator rejects these new
  paired manifests rather than silently reverting to its legacy encoder.
- Fixed two additional in-place quaternion normalizations in the shared
  rotation helper and causal encoder input builder. Mutable and read-only
  input arrays now remain unchanged.
- Added a simulator-only acquisition/return diagnostic using the hash-pinned
  Unitree zero-velocity 29→23 compatibility actor. It is never substituted
  for SONIC during the requested motion and does not emulate Unitree FSM
  ownership, DDS, motor faults or physical recovery.
- Optional standing target projection intersects the existing joint-margin,
  slew and 95% of quarter-effort guard bounds. Empty intersection fails;
  limits are not increased. Every 2 ms substep is observed. Partial failures
  retain terminal state, actual elapsed time and initial tilt rather than
  reporting a false completed interval or zero tilt.

### Correct-pair experiments: limited improvement, all candidates rejected

Native configured simulator gains, full targets, hypothetical 35 Nm ankle
limit, reference start. All are explicit-torque simulator diagnostics; no
claim that these gains are reviewed for this robot.

| Model | 5 rad/s slew completion | No-slew completion | No-slew max pelvis error | No-slew max relative-body error | No-slew max joint RMSE |
|---|---:|---:|---:|---:|---:|
| Original breadth25, correct encoder | 220/535 | 535/535 | 3.8686 m | 0.5462 m | 0.5748 rad |
| Refit residual +0.25, correct encoder | 210/535 | 535/535 | 1.6413 m | 0.6542 m | 0.4825 rad |

The residual improves pelvis and joint errors in this unslewed case but
worsens relative-body error, orientation error (0.4970→0.5760 rad), and
slew-limited completion. Both exceed the predicted quarter-effort guard at
the first unslewed transition; target-margin guards also cross. **Neither
passes motion fidelity or lifecycle screening.** Lower fit residual error is
not evidence of improved closed-loop control.

Four residual alphas were fitted: 0.01, 0.05, 0.10, 0.25. Their slew-limited
completions are 141, 184, 178, 210; all four unslewed cases complete 535 but
fail fidelity. No checkpoint selected or promoted; no original-v14 parity
claim. Earlier stage-one-trained breadth50 with the corrected encoder also
fails every tested fidelity case. Correct pairing is necessary, not sufficient.

Final-code replays are `paired_encoder_20260905_v2/final_baseline/summary.json`
and `final_residual_025/summary.json`; four-candidate fit and initial screens
are `residual_fit/manifest.json` and `residual_eval/*/summary.json` beside them.
All old experiments are preserved, including now-known mismatched-pair tests.

### Acquisition progresses; full dance/return still fail

`paired_encoder_20260905_v2/final_lifecycle/summary.json` starts from the
historical measured motor/IMU snapshot, with estimated foot contact and no
gantry forces. Effort-projected compatibility standing completes **5 s / 250
policy steps / 2,500 physics steps** within existing numeric guards:
minimum height 0.7827 m, max tilt 0.0234 rad, horizontal drift 0.0284 m,
peak quarter-effort ratio 0.95, 77 joint-substeps projected.

After one rigid XY/yaw reference alignment (no height adjustment or per-frame
recentering), the actual SONIC breadth25 dance fails at **187/535**. Active
effort guard first crosses at transition 4. Diagnostic physics deliberately
continues past this recorded crossing; hardware would not follow that trace.
Attempted return from the already-fallen simulated state fails immediately
on an empty effort/position/slew intersection; starting tilt is 1.1973 rad,
height 0.2111 m. This is not a successful return or safe recovery controller.
All 546 source frames pass native23 body-position/orientation FK consistency;
that does not establish dynamic feasibility.

Next work: train/validate a support-capable native23 full-body policy through
one explicitly reviewed actuation profile and its acquisition/return path,
using the actual paired encoder throughout. The existing hardware profile,
stale encoder pin, live PICO calibration, motor-off latch cause and per-motion
29→23 contact/COM feasibility remain unresolved. Do not retry physical dance
based on this entry. Exact preservation of absent waist/wrist axes is impossible.

Verification: **110 focused tests passed**, including actual MuJoCo replay,
paired encoder/decoder export, source/byte mismatch rejection, fitting-pair
binding, immutable quaternions, partial-step accounting and CLI fail-closed
behavior. Critical Ruff checks passed. Separate legacy step1b suite reports
12 passed / 2 failed: existing target MJCF hash is `38d6b0...` while its old
pin expects `16e304...`; fork-local
`artifacts/external/unitree_rl_mjlab/src/tasks/tracking/tracking_env_cfg.py`
is missing. No hash was changed or fixture fabricated to hide those failures.
Heavy artifacts remain local; verified offline code and this log are committed
separately from the pre-existing physical-control experiments.

## 2026-09-05 continuation: deployed-mechanics training and acquisition

**Still not ready for physical full-body dance or live teleop. No robot
commands or mode changes were sent during this continuation.** The new work
is offline diagnosis, training infrastructure and rejected candidate evidence.

### Controller/training alignment implemented

`train_g1_23dof_mjlab_frozen_lora --actuation-profile stage_one_cpp` now reads
the checked-out C++ profile and binds its raw SHA-256 into training lineage.
It trains through explicit 500 Hz torque PD, 50 Hz policy updates, the exact
safe-target transform once, default-relative action fraction, per-joint
scales, slew, and an explicit quarter-effort/target-margin episode guard.
Previous-action observations retain unscaled safe-native23 semantics.
Each environment seeds its slew state from its actual reset pose, after the
motion command reset. Configs are deep-copied; original articulation unchanged.

This is **not** a complete hardware emulator: DDS, watchdogs, ownership,
motor faults, e-stop and Unitree recovery are absent. Training guard response
is termination at the next 50 Hz boundary, not physical recovery. Current
50 Nm ankle table remains an unverified modeling assumption. Old screened
behavior banks cannot be reused without binding the selected profile hash.
The option is opt-in; existing training defaults and robot settings unchanged.

The reader also supports the older committed pre-taper controller (identity
joint scale). Therefore a clean checkout reads its own 0.10/0.25 profile;
this working tree's uncommitted experimental controller supplies 0.60/5.0.
Do not confuse either with a reviewed full-body profile. Unsupported changes
to the armed predicted-effort expression fail rather than silently retaining
the old 0.25 assumption.

### Acquisition and hold results

The simulator can start from a hash-bound historical read-only motor/IMU
snapshot. Root height is estimated by placing the lowest of eight modeled
foot collision spheres on the ground. World position, base velocity and gantry
loads are not measured; this is an offline hypothesis test, not a robot twin.

- `deployment_acquisition_20260905_v2/summary.json`: selected residual decoder,
  measured start, 0.05 s startup hold, deployed gains/fraction/slew. Both ankle
  ablations fail at **49/535 transitions (0.98 s)**, with identical motion,
  minimum pelvis height 0.2352 m and 1.9066 rad maximum additional knee
  flexion. Raising the ankle limit from 35 to 50 Nm does not fix this case.
- `deployment_standing_hold_20260905_v1/summary.json`: the first **1 s contains
  only sampled-posture hold, no policy action**. Knee kp=4, zero feedforward;
  pelvis falls to 0.1359 m, knee flexion increases 2.6087 rad. Predicted
  hold-effort guard crosses at 0.368 s. Diagnostic physics continues after
  that crossing; actual controller would enter its fault path. This does not
  simulate or disprove Unitree's own standing controller or handback sequence.
- Acquisition v1 is superseded: it reused the old 0.06 m ankle-origin shortcut,
  initially leaving modeled feet above the ground. V2 uses collision geometry.
  Legacy reference/neutral behavior was preserved for numerical comparisons.

### Bounded training attempt: rejected

`stage_one_training_20260905_v1/breadth50/`: seed 20260905, eight-clip original
SONIC/PICO rehearsal corpus, 16 environments, 16 steps/environment, 50 PPO
updates = 12,800 transitions. Frozen SONIC encoder/base decoder plus rank-8
LoRA; 253,944 trainable actor parameters. Lineage:
`3c90acb54b299a968a8230ee211f2a20d5900f59efc4eae44a4dffbe5540f1ba`.
All 30 recorded source-file hashes matched immediately after training.
The historical header hash is
`5c7965251b49d4f3005e9802aebddb03f60534603992c11837078e5ac41badaf`.
Subsequent parser hardening/backward compatibility changes intentionally
change source lineage; do not force an exact resume across that change.

Training finished normally, but final mean episode length was only 17.32
policy steps. Updates 25 and 50 were both materialized, exported and checked
with three ONNX parity cases (maximum absolute errors 2.146e-6 and 1.907e-6).
Both then failed closed-loop happy-dance evaluation at **46/535 reference
start and 49/535 measured start**, using the deployed 0.60/5.0 mechanics with
0.05 s startup hold. All four fidelity screens fail. Neither candidate replaces
the selected baseline, grants promotion, or establishes improvement/parity.
Artifacts are under `breadth50/eval/model_{25,50}_envelope/summary.json`.

Paths above are relative to `artifacts/g1_true23_frozen_lora/`. Heavy model and
telemetry artifacts remain local. The first `smoke/` run was a plumbing check
while source edits were ongoing, not exact-lineage candidate evidence.
The final `smoke_final/` rerun passed with all 30 current source hashes matching;
its lineage is `995c58e30c180b9669d5f1b805917c59c378a0b9b26a1ddc41b6abb9eea56cbb`.
The actual committed pre-taper header also passed the reader compatibility
check without modifying the current experimental header.

Verification: **95 focused Python tests passed**, including actual MuJoCo
numerical equivalence, foot-contact placement, malformed telemetry rejection,
Torch/NumPy PD agreement, configuration isolation, frozen artifact validation,
and training lineage tests. Critical Ruff checks passed. Physical controller,
launcher and widened-threshold edits remain separate from the offline commit.

Next: establish a physically support-capable, hardware-reviewed actuation and
acquisition/return controller; train and screen against that same controller.
The short failed run does not prove learning is impossible, but offers no
reason to retry the robot. PICO remains unqualified; 29→23 retargeting still
requires per-motion contact/COM feasibility and cannot preserve absent joints.

## 2026-09-05 follow-up: readiness correction

**Physical full-body dance/live teleop is NOT ready.** This follow-up sent no
LowCmd or locomotion-mode commands. Historical entries below are preserved,
including earlier conclusions now contradicted by stronger evidence.

### What the new checks establish

- Read the complete paginated transfer-chat history and the original
  `23dofsonic` user/final turn history; traced the current training, native23
  mapping, selected decoder, replay, active controller, handoff and tests.
- Current robot is reachable. Saved read-only probe at 10:36 UTC reports
  3,389 CRC-valid samples, zero invalid samples, all 23 controlled motors in
  mode 1 with motorstate 0, knees 0.298/0.313 rad, IMU roll/pitch
  -0.007/0.011 rad. This establishes enabled telemetry, not load-bearing
  balance. Evidence: `artifacts/g1_true23_frozen_lora/readiness_audit_20260905_v1/motor_health.json`.
- The pinned XRT binding exists in the ORIGINAL checkout and imports correctly
  through explicit `PYTHONPATH`. The fork's default binding directory is empty.
  A fresh read-only PICO probe produced 60 snapshots, zero connected trackers,
  no calibration and `passed=false`. Service was not running; this does not
  prove the headset itself is powered off. Evidence: same directory,
  `pico_health.json`. Do not label saved packets a live-headset test.
- Deployed knee kp is 16; pre-arm/return hold knee kp is only 4. The legacy
  simulation used knee kp 99.0984. Deployed target fraction is 0.60 with
  5 rad/s slew; legacy simulation used full targets with no slew. These are
  different controllers, not a small deployment detail.

### Implemented corrections

1. Orientation-error diagnostics no longer normalize views into live MuJoCo
   state or source reference in place. Regression tests require unchanged
   inputs, including read-only arrays. This numerical mutation explained the
   inconsistent reproduction: the corrected full-authority legacy case now
   reproduces its historical 535/535 and metrics.
2. Separate reference-relative motion-fidelity screen from legacy survival.
   Full clips required; missing/nonfinite metrics fail. Provisional maximum
   errors: pelvis 0.25 m, orientation 0.5 rad, relative tracked bodies 0.20 m,
   joint RMSE 0.35 rad. These are engineering screening criteria, NOT physical
   limits, demonstrated perceptual parity or deployment authorization.
3. Direct standing-dance terminal evidence now includes first/last armed
   joint arrays and independently recomputed bilateral knee flexion. Both
   C++ and Python reject missing/nonfinite/inconsistent evidence and net
   bilateral flexion over 0.5 rad. The observed 1.24 rad sag cannot be called
   successful just because joints moved. This conservative endpoint screen
   runs AFTER recovery; it is not a real-time fall detector, does not prove
   choreography, and is not applied to arbitrary live crouch motions.
4. Fixed cross-platform Windows-to-WSL path conversion. Added exclusive
   output files for motor and native-runtime read-only probes.
5. Preserved and tested earlier enabled-motor checks and live-launcher CPU
   argument repair. Fixed separate FSM utility leaking into the main CMake
   source glob. Removed contradictory taper comments; no control gain,
   action, joint, effort or watchdog limits were increased in this follow-up.

### Fresh simulation result (after diagnostic mutation fix)

`artifacts/g1_true23_frozen_lora/deployment_envelope_20260905_v3/summary.json`
contains 16 cases at a hypothetical 35 Nm ankle model limit, both reference
and default-neutral starts, stage/released gains, 0.60/1.0 fraction, 5/no slew.

- Stage gains: all eight cases fail, completing 41–53 of 535 transitions.
- Released gains + full target + no slew + reference start: 535/535 under
  legacy gates, but pelvis position error **5.0104 m**, relative-body error
  **0.6072 m**, joint RMSE **0.7170 rad**. Predicted quarter-effort guard is
  exceeded at transition 0 by a factor of 3.066; target-margin guard crosses
  at transition 106. These guard crossings are recorded, not enforced by the
  diagnostic physics loop. This is NOT a deployable replacement profile.
- The same released/full/no-slew controller from default-neutral start fails
  at 274/535. Acquisition cannot be assumed from a reference-state replay.
- All 16 fail the provisional fidelity screen. The independent immutable
  audit of 29 older cases also has zero fidelity passes; historical originals
  remain untouched in `readiness_audit_20260905_v1/fidelity_audit.json`.

### Corrections to earlier explanations

“Amplitude 1.0 will fix balance” is not established: it fails under deployed
gains. “All training configs use 50 Nm, so hardware 35 Nm is wrong” is not a
valid hardware-rating argument. Unitree's current training asset explicitly
models the parallel ankle linkage using a nominal 1:1 approximation because
its exact geometry is unknown; it doubles the individual 25 Nm model value.
That is a modeling assumption, not a verified configuration-dependent torque
envelope for this particular robot:
[official model source](https://github.com/unitreerobotics/unitree_rl_mjlab/blob/main/src/assets/robots/unitree_g1/g1_constants.py).

Current physical effort/tolerance changes from the earlier session are left
intact for review, not endorsed or newly deployed. Normal handback code may
itself request the Unitree damp FSM as an intermediate transition: “zero
damping LowCmd packets” never meant “robot never enters damping.” The legacy
`g1_restore_walkrun` helper still uses FSM500 and RPC-only success evidence;
it is NOT a qualified standing-restoration test. It was not run here.

### Next work, in order

1. Establish one explicit hardware-reviewed actuation profile: kp/kd,
   default pose, action transform, slew, motor/joint effort limits including
   coupled ankles. Simulate that exact profile, including acquisition from
   measured FSM801 posture and the return-to-Unitree interval. Do not blindly
   copy high gains or weaken guards to reproduce legacy completion.
2. Adapt/fine-tune the frozen SONIC decoder against this actual profile and
   native23 contact/hand/COM references. Preserve known walk/standing clips;
   reject failed teacher windows. New candidates need full-clip fidelity,
   disturbances, and preservation comparison against original v14.
3. Qualify each 29→23 retargeted motion separately. Missing waist roll/pitch
   and bilateral wrist pitch/yaw make exact reproduction of every 29-DoF move
   impossible. Contact-preserving retargeting, tempo reduction and explicit
   rejection of infeasible clips are required; output-row selection is not
   morphology compensation. No replacement of full SONIC by upper-body-only
   control is claimed.
4. Restore calibrated live PICO tracking and qualify PICO→MuJoCo first. Only
   after profile/trajectory qualification, a fresh shadow, and an
   operator-supervised gantry test can physical teleop be called ready.

No new checkpoint, promotion, robot actuation, commit or push was performed
in this follow-up. Source remains in this separate working tree. Goal remains
unfinished; evidence/launcher fixes are not a balance-policy fix.

Verification: 132 focused Python tests passed, including an actual full-length
MuJoCo numerical-equivalence replay. The C++ active controller and dependency-free
core harness built; the harness passed nine robot-free lifecycle scenarios
and 4,000 recovery frames with zero published damping frames. Ruff critical
error checks and `git diff --check` passed. Root `ctest` discovers no tests in
this existing build layout, so the harness was run directly; no empty CTest
result is counted as a pass. Native runtime inspection passes saved-clip
shadow prerequisites but explicitly returns `physical_teleop_ready=false`;
its evidence is `readiness_audit_20260905_v1/runtime_preflight.json`.

Session date: 2026-09-03 → 2026-09-05.
Robot: Unitree G1, 23-DoF, on gantry. Control board `192.168.123.161`,
onboard Jetson `192.168.123.164`.

---

## 1. Executive summary

The starting complaint was "the robot goes limp as soon as the dance starts,"
and later "I have never seen the dance at all."

Both turned out to be true, and neither had a single cause. Five distinct
defects were found and fixed, one of them a genuine sim-to-real mismatch that
explains why the policy worked in simulation and faulted on hardware.

| # | Problem | Status |
|---|---|---|
| 1 | Limp at startup | Was already fixed before this session; verified, not touched |
| 2 | Limp at handback | **Fixed** — `SetFsmId(801)` was never issued anywhere |
| 3 | Dance invisible (0.003 rad/s of motion) | **Fixed** — two throttles, speed cap dominant |
| 4 | Ankle effort faults on every run | **Root-caused** — sim2real limit mismatch, fixed |
| 5 | Robot falls when actually bearing weight | **Identified, not fixed** — needs full policy authority |

---

## 2. Root cause of the sim2real gap

The policy was trained against an ankle `effort_limit: 50.0`. The deploy
controller enforced `35.0`.

Verified across **all 25 training runs** under
`external_dependencies/unitree_rl_mjlab/logs/rsl_rl/g1_23dof_tracking`,
including the `actuator_safe` v7/v8 variants. Every other actuator group
matched the controller exactly:

| actuator group | training `env.yaml` | controller `kHardwareEffortLimitNm` |
|---|---|---|
| shoulders, wrists | 25.0 | 25.0 |
| hip pitch/yaw, waist | 88.0 | 88.0 |
| hip roll, knee | 139.0 | 139.0 |
| **ankle pitch/roll** | **50.0** | **35.0** |

Joint *ranges* matched sim-to-real perfectly, so this was never a kinematics
problem. On hardware the ankle asked for 54.7 Nm — essentially what it was
trained to use, and 56% over what the controller permitted.

This explains the entire symptom pattern: arms, hips and knees always
transferred cleanly and only the ankles ever faulted, always at 35–55 Nm.

The G1 ankle is driven by a parallel actuator pair, so per-joint capability
plausibly exceeds the naive single-motor URDF figure. The controller table was
aligned to 50.0 on the strength of the training config being the deliberate,
campaign-wide value.

**Conclusion: no redesign needed.** SONIC transfers. 7 of 8 motions pass in
simulation with the same 23-DoF decoder (walk001, walk010, pico_upright,
pico_standing, elbow_crawl, hand_crawl, happy_dance; only pico_crouch fails on
a crawl physical gate).

---

## 3. Why the dance was invisible

Three throttles, found in order. The first two are the reason 5163 policy
frames produced motion the operator could not see.

1. **Speed cap was dominant.** `kStageOneTargetRateRadPerSecond = 0.25`
   saturated the slew limiter on *every* 2 ms step and clipped **47.9%** of all
   joint-steps. The source choreography needs p95 = 4.42 rad/s, max 13.45.
2. **Amplitude at 10%.** `kStageOneActionFraction = 0.10`.
   Combined result: ~0.003 rad/s of joint motion, roughly a tenth of a degree
   per second. Measured per-second joint range was 0.002–0.04 rad.
3. **Per-joint ankle tapers broke coordination.** Scaling ankle pitch/roll to
   0.18–0.70 while hips/knees ran at 1.0 passed the policy's disturbances
   through at full strength while attenuating its corrections. Removed; scaling
   is now uniform only.

After fixing 1 and 2, per-second joint range went to 0.10–0.54 rad — roughly
25× more motion, and visible.

---

## 4. Controller changes

All in `gear_sonic_deploy/src/g1/g1_deploy_onnx_ref/`.

### Handback (the original "goes limp" bug)
`SetFsmId(801)` was never issued anywhere in the codebase. `g1_restore_walkrun`
goes 1 → 4 → 500, and FSM 500 never engages with the feet unloaded, so recovery
stopped at the FSM 4 crouch and reported success.

- **FSM 4 is a crouched stand**, not a finished one — its stand-up transition
  ends with knees near 0.63 rad, not the ~0.29 rad of a real stand.
- Handback now drives `damp → FSM 4 → wait for fsm_mode 0 → FSM 801`.
- **Ends at 801, not 500.** FSM 500 (walk-ready, balance-armed) decayed to zero
  torque after minutes on a gantry and then refused further FSM commands. 801
  held for a full 8-minute idle watch with zero change.

### Overload protection (replaces flat effort limit)
A single instantaneous threshold cannot distinguish a 6 ms dynamic peak from a
sustained stall. Now:
- absolute ceiling at `1.2 ×` continuous → instant latch;
- between continuous and ceiling, integrate the excess against a 4.0 N·m·s
  budget with 8.0 N·m/s recovery (I²t style);
- 3-consecutive-sample debounce on the raw comparison.

### Measured-position tolerance (0.02 rad)
`L_ankR` faulted at 0.26186 against a 0.26180 limit — 0.003° over. The joint
rests on its mechanical stop and the encoder reads a hair past it. Proven not to
be command-driven: with the ankle-roll command zeroed entirely, the joint still
reached the stop, so body reaction drives it there.

### Robustness: degrade instead of abort
- **Armed causal-join gaps** (`lowstate_coverage_timeout`) now skip the frame
  and hold the last target, bounded at 10 consecutive, instead of tearing down a
  live session. The policy-freshness watchdog remains the real guard against
  genuine input loss.
- **Restore retries `CheckMode`** 20× over 10 s. A single failed RPC previously
  left the motion service released and the robot limp in external-control limbo,
  requiring hands-on recovery.
- **Pre-arm hold retries** up to 50× at 10 ms waiting for a fresh policy instead
  of aborting on one stale read. Same freshness requirement still enforced.

### Evidence and gating
- `fault_joint` / `fault_value` recorded for both effort and position faults.
  This is what identified the ankles instead of guessing.
- `measured_armed_excursion_rad` — largest measured joint travel while armed,
  enforced ≥ 0.03 rad by the Python validator. A run had reported `passed=true`
  on a robot whose motors were disabled and which never moved; that is now
  impossible, and the old evidence was replayed through the new validator to
  confirm it is rejected.
- `armed_knee_flexion_delta_rad` — net knee flexion from first to last armed
  sample. Excursion alone cannot distinguish dancing from collapsing, because a
  collapse is large travel too.
- `armed_causal_join_skips`, `restore_check_mode_retries`,
  `pre_arm_hold_prepare_attempts` for the robustness paths.

### Envelope
```
kStageOneActionFraction            0.60    (was 0.10)
kStageOneTargetRateRadPerSecond    5.0     (was 0.25)
kHardwareEffortLimitNm[ankle]      50.0    (was 35.0)
kStageOneJointAmplitudeScale       all 1.0 (per-joint tapers removed)
kMaximumDirectDancePostArmSeconds  11      (was 5; routine is ~10.7 s)
```
Envelope values are mirrored in
`gear_sonic/scripts/authorize_g1_true23_frozen_lora_dance_gantry.py` and
verified against the compiled constants at runtime, so both must change together.

### Harness tests updated
- effort test rewritten for the overload contract (brief peak tolerated,
  above-ceiling latches instantly, sustained overload exhausts the budget);
- slew test threshold now derived from the slew constants instead of a
  hardcoded `0.049` that was only valid at the old rate.

`true23_active_gantry_core_harness` and
`qualify_g1_true23_active_lifecycle_no_robot` pass after every change.

---

## 5. New tooling

- **`g1_fsm_command`** (new binary, `src/g1_fsm_command.cpp` + CMake target) —
  issues an explicit locomotion FSM id or balance mode and reports before/after
  state. Read-only otherwise; never opens LowCmd. This is what made
  software-only recovery possible.
- **`lowstate_torque.py`** (scratchpad) — read-only per-joint torque, mode and
  posture probe with IGMP membership handling. The authoritative "is it actually
  standing" check.
- **`record_traj.py`** (scratchpad) — high-rate joint + IMU recorder.
- **`render_skeleton.py`** (scratchpad) — replays a recording through MuJoCo
  forward kinematics and renders an mp4 of the actual motion. Used to show the
  operator what the robot really did.
- **`recover.sh`**, **`auto_loop.sh`**, **`idle_watch.sh`** (scratchpad) —
  recovery sequence, autonomous repeatability loop with self-recovery and a hard
  stop, and an idle-decay watcher.

---

## 6. Onboard aarch64 build

Native build works on the Jetson (`192.168.123.164`, user `unitree`).
All five binaries built; all three safety harnesses pass **on the robot**:

```
true23_shadow_gate_harness         all checks passed
true23_live_shadow_core_harness    PASS
true23_active_gantry_core_harness  all checks passed;
    robot_free_lifecycle_scenarios=9  recovery_frames=4000
    published_damping_frames=0  dds_opened=false
```

What it took:
- **g++-10 via dpkg.** Ubuntu 20.04 ships gcc 9.4, which lacks `<span>` and
  `std::jthread`. `apt` is blocked by a pre-existing chrony/systemd-timesyncd
  conflict, so packages were downloaded and `dpkg -i`'d directly — this avoids
  the resolver and does not touch the robot's time sync.
  `libgcc-10-dev` / `libstdc++-10-dev` remain `iU` (unpacked, unconfigured)
  because of that same conflict; files are present and the build works.
- Vendored aarch64 ONNX Runtime (already in
  `external_dependencies/unitree_rl_mjlab/deploy/thirdparty/onnxruntime-linux-aarch64-1.22.0`).
- msgpack + nlohmann headers copied from the workstation (no apt).
- **Repair of Windows-checkout SONAME symlinks** — `.so` / `.so.0` entries are
  19–24 byte text stubs after a Windows checkout and must be recreated as real
  symlinks.

Not yet done: an actual onboard *run*. The build and safety cores are verified
on the robot; the control loop has not been executed there.

---

## 7. Test results

### Repeatability loop, amplitude 0.6, gantry tight
| run | outcome | excursion | restored |
|---|---|---|---|
| 1 | pre-arm abort (failed closed, robot untouched) | — | — |
| 2 | PASS, full 11 s | 1.273 rad | 801 |
| 3 | shadow failed before controller start | — | — |
| 4 | PASS, full 11 s | 1.200 rad | 801 |
| 5 | PASS, full 11 s | 1.215 rad | 801 |
| 6 | stopped 5.81 s, no fault | 1.095 rad | 801 |

3 clean full-length passes out of 4 runs that reached the robot. The two
non-runs failed closed without touching it.

**Run 6 proved both robustness fixes in the field**: `skips=1` (a causal-join
gap absorbed instead of aborting) and `retries=1` (a failed restore `CheckMode`
recovered, ending at 801 instead of stranding the robot limp — exactly the
failure that previously required a control-board power cycle).

### After the gantry was loosened
First run at 0.6 produced the largest motion of the session **and collapsed**:

```
measured_armed_excursion_rad   1.491 rad (85°)
armed_knee_flexion_delta_rad   1.240 rad (71°)   ← collapse
final_fault  joint_position_limit, L_ankR at -0.282
inference_error  pico_age_out_of_range, age 360 ms
restore_check_mode_retries  3   (still landed upright at 801)
```

The collapse detector added earlier that night is what caught this. Excursion
alone scored it as the best run of the session.

---

## 8. The gantry investigation

This went through two readings; both are recorded because the correction
matters.

**Initial measurement (gantry tight), standing at FSM 801:**
```
L_knee  -9.14 (std 0.087)    R_knee  -6.81 (std 0.131)
L_ankP  -5.32 (std 0.084)    R_ankP  +3.02 (std 0.135)
IMU pitch -0.23°, std 0.021°
```
Ankles pushing in *opposite* directions, torques essentially frozen, no
postural sway. Read at the time as "the gantry is carrying the robot."

**After the operator loosened it:**
```
L_ankP  +6.14    R_ankP  +9.11     ← same sign now
L_knee  -6.43    R_knee  +0.81
```
Ankle load roughly doubled and the sign conflict disappeared — consistent with
the feet taking weight and the harness no longer twisting the legs.

**After a second, larger loosening:** numbers identical to two decimals
(L_ankP 6.17, R_ankP 9.12, R_knee 0.81). No further load transferred, i.e. the
straps were no longer bearing any.

**Corrected reading:** the first loosening did transfer load to the feet. The
low-sway criterion (invented as `>0.1°` without a baseline) was not a reliable
indicator — Unitree's FSM 801 hold is simply very stiff. The decisive evidence
was that the robot then *collapsed* under the dance, which only happens if it is
genuinely supporting itself.

**Consequence:** all results collected before the loosening validated the
pipeline — timing, gates, handback, fault handling — but said nothing about
balance, because the robot never had to balance.

Residual asymmetry worth checking: `L_knee −6.4` vs `R_knee +0.8` suggests it
stands slightly lopsided, favouring the left leg.

---

## 9. Known-good operating procedure

1. Stand: `g1_fsm_command eth0 fsm 801 I_CONFIRM_G1_FSM_COMMAND`
2. Verify by **posture, not FSM id**: knees ≈ 0.285 / 0.329 rad, motors 23/29
   live, torque holding.
3. Dance: `DUR=11` through the chain script.
4. After re-selecting the `ai` service, wait ~8 s — FSM commands fail with error
   3104 until the service initialises.
5. Recovery from limp: `damp → FSM 4 → wait fsm_mode 0 → FSM 801`, ~14 s,
   no remote needed.

---

## 10. Open items

- **Full policy authority (1.0) not attempted.** A balance controller is only
  self-consistent at full authority; 0.6 is the highest tested and it collapses
  once the robot bears its own weight. This is the next test and needs a human
  at the e-stop.
- **Host contention** — PICO packet ages of 90–360 ms against a 100 ms limit,
  costing 1–2 retries per launch and causing mid-dance faults. The onboard build
  is the real fix.
- **PICO teleop input entirely unproven.** No headset, trackers, calibration or
  body stream has ever connected. The XRT pybind module is not present on this
  workstation and the PC service is not running.
- **Teleop needs the control loop decoupled from the input stream.** The
  causal-join fix applies this principle in one place; live teleop needs it
  throughout — late headset packets must degrade tracking, never end a session.
- **Free-standing not authorized** (`free_standing_authorized: false` in the
  promotion sidecar) and unvalidated. Not a flag to flip — a reviewed promotion.
- **Latched motors-off, twice.** After heavy runs the drivers latched off
  (`motorstate` bit 30 set, `mode` 0 on all joints) and cleared only on a
  control-board power cycle. Battery 52.5 V and motors 35 °C at the time, so
  neither power nor thermal. Root cause unknown; worth raising with Unitree.
- **Only one motion is promoted to hardware.** There is exactly one causal
  packet bundle (`original_sonic_happy.true23.causal_packets.json`), pinned by
  hash. Running walk001 or elbow_crawl on the robot needs a bundle generated per
  motion plus a new reviewed promotion.
- **Session ended with WSL having dropped `eth0`** (recurring mirrored-networking
  glitch, fixed previously by `wsl --shutdown`). Robot state at that moment was
  unverified; last confirmed reading was standing at 801, knees 0.291/0.339,
  23/29 motors live, holding.

---

## 11. Verification discipline

This is the most transferable lesson from the session.

FSM id, exit codes and evidence flags repeatedly reported success while the
robot was limp, motionless, or collapsed. Specific traps hit:

- `g1_mode_probe` returned `ai / 801 / 0` while the robot hung dead in the
  straps. FSM id says nothing about torque.
- A full 11 s run reported `passed=true`, `motion_mode_restored=true`, 100 stable
  restore samples — with every motor disabled and zero movement.
- `measured_armed_excursion_rad` of 1.49 rad looked like the best run of the
  session; it was a fall.

Trust only:
- motor `mode` counts (23/29 expected; 13, 14, 20, 21, 27, 28 are absent on the
  23-DoF robot and correctly read 0);
- `tau_est` magnitudes;
- joint angles against the standing reference (knees 0.285 / 0.329);
- IMU tilt;
- `armed_knee_flexion_delta_rad` for collapse.

Joint index mapping matters: compact 0–12 map identity to motor slots, but
compact 13–22 (the arms) map to slots 15–19 and 22–26. An early per-joint
analysis read the wrong slots and produced misleading torque figures.
