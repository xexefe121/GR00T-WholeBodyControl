# Native23 full-body VR teleop — current priority

CURRENT: compact native23 reference-residual learner trial started(main32479,
train2000_v2,driver_v4). Earlier main73693 stopped on a numerical audit error
after5updates, preserved. Derived float32 summation bound fixes that audit;
returned/stored rewards were exact. New v4 smoke and input audit both pass.
Same PICO/SONIC motion inputs and physical limits; distinct165-input/180014-
parameter tracker, not another frozen-decoder adapter. Smoke384transitions and
independent16-state input/optimizer audit pass. Maximum continuous12.288m
transitions; first full-replay gate1.2288m. No deployment claim from smoke.
Untrained tracker fails all4requests during standing at56controls. No robot
commands. Goal ACTIVE; see newest PROGRESS.md entry and compact_tracker artifacts.
Older CURRENT statements below are historical, not current job status.

CURRENT: all new controller tests complete; none deployable. Bounded ankle
torque repairs original30.60s local PICO stop, then whole-body balance fails
at33.08/115.60s. Pelvis drops to11cm while target remains73cm. All4 motion
tracking screens fail. Independent dynamics, torque arithmetic and original-
prefix audits pass; they are not motion success. No job still running, no
hardware path changed, no checkpoint exported. Goal ACTIVE; NOT READY.
Latest result:artifacts/g1_true23_pico_ankle_feedforward_20260910_v1/OUTCOME.md.
Next: measured native23 learning/balance objective audit, not more guard tweaks.
Entries below preserve earlier experiment states and are historical.

Latest target-only100ms controller REJECTED: PICO31.90/115.60s, all4 tracking
screens fail, filter exceeds20ms on252 controls. Full independent replay and
prediction audits pass; they do not turn those results into readiness.
New fixed SIM-only branch tests bounded ankle torque after original20ms target
search fails, clipping TOTAL torque within unchanged native effort limits.
Copied-state2.5Nm correction works at final1000's old failure but not parent500's
later failure.38 tests pass; complete lifecycles being evaluated separately.
No physical controller path, checkpoint promotion, gain/limit change or training.
Goal ACTIVE; deployment NOT READY. See latest PROGRESS.md entry.

Current bounded SIM test: anticipatory ankle braking. Exact saved-state force
analysis finds contact-loaded ankle near its range boundary with no meaningful
additional inward PD-target authority; old search retries nearly the same
target8times. New separate filter adds100ms constant-target ankle checks while
retaining every-joint20ms checks, native bounds/gains and all original source
inputs.27 tests pass; saved-state/full-lifecycle results pending. No automatic
reward-only training extension. Full-body goal ACTIVE; deployment NOT READY.
See artifacts/g1_true23_pico_ankle_horizon_20260910_v1/EXPERIMENT.md. Robot untouched.

Latest experiment COMPLETE, not stalled: independent foot-placement reward and
validated learner continuation from PICO500. Main94600 finished500→1000 with
1,024,000 new transitions; all stored rewards/optimizer continuation audited.
Both full600/1000 evaluations and independent physics replays finished. All
eight tracking screens FAIL. PICO600 reaches75.14/115.60s; final1000 regresses
to30.60s, despite better first30.60s root/foot errors than parent500. Walking
003/008 also stop early at1000. No checkpoint promoted or automatic extension.
19,968 sampled lower-body/VR reference inputs are exact; existing observation
noise explains the retained failed noiseless orientation audit. No encoder
replacement, lost joints, changed limits or physical robot operation.
Full-body goal ACTIVE; deployment NOT READY. All experiment jobs terminal.
See PROGRESS.md and artifacts/g1_true23_pico_foot_precision_20260910_v1/OUTCOME.md.

Current work: existing-PICO SONIC adaptation evaluated, not waiting for a
headset. New lossless bank contains walk002/003/PICO;008 stays outside optimizer.
Two-update GPU wiring smoke and independent reward/weight audit pass. Complete
100-checkpoint requests fail all four full-body tracking screens; PICO stops
after30.48/115.60s. Independent saved-torque and policy replay checks are exact,
but do not turn failed motion into readiness. Fixed continuous500-update run is
complete:500 updates/1,024,000 transitions. All500 walking timelines complete
but fail tracking; PICO improves to69.34/115.60s then the range guard rejects.
Matched50.60s root p95 improves5.239931→0.359780m, but leg/foot fidelity still
fails. The complete69.34s candidate root p95 is1.053879m. Independent final
physics/policy audit passes; no checkpoint promoted. That original run is
complete; the new foot-objective continuation above is active.
Existing native124 shortcut failed all eight comparisons and is not
a teacher or replacement. This is **NOT deployment-ready**. See latest entries
in PROGRESS.md and artifacts/g1_true23_pico_training_20260910_v1/OUTCOME.md.

User redirected work on 2026-09-10: **VR readiness, not dance readiness.**
Do not resume dance/reference-fitting campaigns as the next goal step. Full-body
remains the requirement; an upper-body-only controller is not a substitute.
No physical robot operation is authorized by this document.

Latest instruction explicitly requests **a new full-body PICO teleop goal using
only the existing recorded clips, including legs**. This supersedes waiting for a headset as
the prerequisite for the current saved-input simulation work. Live PICO capture
is still a separate untested deployment requirement, not something a recording
can prove. Do not resume unrelated arbitrary-dance fitting campaigns.

**New goal ACTIVE (2026-09-10).** The latest goal-state read returned no existing
goal; creation of the user's new goal succeeded. Earlier duplicate-goal failures
remain historical, not the current blocker. No old objective was marked complete.

Goal: make SONIC full-body teleop work in native23 simulation using existing
PICO/PICO-derived recordings, then deliver a tested configuration and procedure
for separately supervised hardware validation. Acceptance includes complete
motion tracking, both legs and arms, source timing, joint/effort limits, standing
entry/return and simulated pause/disconnect/recovery. Preserve failed evidence;
compare against the measured baseline. A connected headset is not required for
this work. Saved clips cannot qualify live sensor capture or physical handback.

## Latest runtime result — partial improvement, still not deployable

Prepared reference-only kinematics removes unnecessary dynamics work with
identical fields across1817 existing packets. Non-spinning ONNX worker settings
preserve all656 captured policy outputs and complete walk→balance physics.
Fixed actual-clock matrices improve3/6→5/6 scenario passes. Two full walks pass;
third stops after212/656 SONIC controls because process wakes18.67ms late.
All20ms deadlines,100ms freshness and original physical limits stay unchanged.

Independent fresh torque replay reproduces108,720 substeps exactly across all
twelve attempts. Range/effort checks pass; this does not repair unchanged
leg RMSE0.207371rad or26.61/20.83cm relative foot errors. Timing AND motion
fidelity remain blockers. Evidence:
`artifacts/g1_true23_reference_runtime_20260910_v1/OUTCOME.md`.

New opt-in SIM receiver: `gear_sonic.scripts.run_g1_true23_prepared_paced_sim`.
It requires full source completion for exit0, not merely a successful balance
tail after input/timing failure. No hardware route or live-headset qualification.

## Latest controller result — both momentum variants rejected

After-inference internal momentum correction advances the full PICO request to
32.60/115.60s before the native ankle guard rejects. Same25.58s prefix leg RMSE
improves0.124231→0.119676rad; root p950.228086→0.218382m. Relative feet remain
9.90/11.56cm. This is not full-source or full-body tracking qualification.

Moving the observer update before inference reaches33.04s, then crosses an
internal missing-axis bound. Common-prefix root error worsens to31.01cm and
leg RMSE0.120386rad. Both candidates rejected. Independent native physics,
history and network audits are exact; physical success is not inferred from
code parity. Coordinated one-step range search is also not a live solution:
500ms compute and>2rad target change. No controller promoted.

Evidence: `artifacts/g1_true23_momentum_update_20260910_v1/OUTCOME.md` and
`artifacts/g1_true23_current_observer_20260910_v1/OUTCOME.md`.

Separate runtime correction results are above. This is not a tracking fix.
Goal ACTIVE. Full-body physical VR remains NOT READY; robot untouched.

## Earlier tracking experiment — partial gains, full trial rejected

SIM-only internal source29 model retains six explicitly predicted missing-axis
q/dq/action channels alongside unchanged measured native23 feedback. All23 real
actuators and physical limits remain unchanged. This is NOT a hardware observer
or promoted teleop runtime. The preceding affine model failed three walking
checks and was not used for a native trial.

New native PICO trial completes25.58/115.60s source, then stops on INTERNAL
waist-roll range failure. Same-prefix root p95 improves0.813480→0.228086m and
arm joint RMSE0.295870→0.261937rad. Leg RMSE0.124182→0.124231rad is unchanged;
pelvis-centered foot p95 improves to9.75/12.36cm, still poor. World feet remain
22.54/23.11cm against unchanged5cm screens. Some hand-point metrics worsen.
No full motion, return, real-time or deployment pass is claimed.

26 focused tests pass. Independent native saved-target physics matches all
16,290 substeps exactly; actual joint/effort excesses0. Independent history/model
audit and96 frozen-source reinferences are exact. The source-side predictor
preflight alone reproduced all four source recordings but did not qualify their
original tracking. Evidence and rejected-outcome details:
`artifacts/g1_true23_virtual_state_20260910_v1/OUTCOME.md`.

Do not relax the virtual stop or physical limits to turn this rejection green.
Next controller work must address measured virtual waist oscillation and actual
leg/foot fidelity; another unchanged training or target-clamp sweep is not justified.
Goal remains ACTIVE. No robot commands, hardware damping diagnosis or promotion.

## Fixed-rate input-loss simulation — verified, not a tracking fix

Latest addition goes beyond virtual timing: an actual50Hz localhost publisher/
consumer now runs existing saved packets with a40ms startup buffer and records
all controller deadlines and physics substeps. A startup ordering defect was
fixed: compiled-model hashing took231–291ms and now finishes before subscribing.
The100ms freshness check and20ms compute budget were not relaxed.

Actual pause, dropped-packet and malformed-JSON cases each pass their limited
fault screens:200 SONIC controls,706 balance controls,18.12s wall time, no
deadline misses. Returned packets cannot silently restart SONIC. Independent
checks verify36,240 substeps across all four paced cases with no joint/effort
excess and motor velocity ratios<=0.5693.

**Uninterrupted real-time walking FAILS:** one25.336ms control exceeded20ms at
control393, latching balance after394 SONIC controls. The remaining512 controls
were balance, not source motion. All656 source messages arrived, so this is not
a demonstrated packet-loss problem. Exact executed SONIC prefixes and a new
full virtual replay match the old failed-tracking baseline; no fidelity gain.
Slower alternative CPU inference settings were measured and rejected.

New additive entry point: `gear_sonic.scripts.run_g1_true23_paced_sim`.
Reproducible saved-source harness, failures, measurements and boundaries:
`artifacts/g1_true23_vr_paced_20260910_v1/OUTCOME.md`.
The older live receiver is unchanged; the new path is diagnostic, not a promoted
deployment route. Timing outliers, tracking, standing acquisition, explicit
SONIC re-entry, live headset and physical handback remain unqualified.

New `gear_sonic.scripts.record_g1_true23_clocked_input_sim` runs one controller
tick every20 ms of virtual time, including missing-input intervals. It latches
the existing balance actor on the first missing/invalid input deadline and
records every500-Hz physics substep. No implicit re-entry on returned packets.
This is additive SIM-only code; the older blocking live consumer is unchanged.

Six cases on existing walk002 completed: nominal,500-ms mid-walk pause, sequence
gap, malformed input, stale startup and full walking followed by5 seconds of
EOF balance. Independent verification of41,860 substeps confirms zero measured
joint-range excess, commanded effort within limits and motor velocity ratios
<=0.5693.27 focused tests pass. EOF ends with maximum joint speed0.006822rad/s.
The nominal trace is bit-exact to the failed-tracking baseline below: no new
policy improvement. Fault cases stop source tracking and latch balance; they
are not successful completion of the remaining motion or SONIC reacquisition.

Evidence and reproducible command:
`artifacts/g1_true23_vr_clocked_20260910_v1/OUTCOME.md`.
Measured full walking-plus-balance video:
`artifacts/g1_true23_vr_clocked_20260910_v1/end-of-stream/walk_then_balance.measured.mp4`.
Virtual-time testing does not prove actual receiver latency, real-time deadlines,
normal-standing acquisition, SONIC re-entry or physical firmware handback.

## Public walking tracking baseline — simulation, not deployment ready

Fresh public TWIST2 walk002 replay completed all **656 controls / 13.12 s** with
the validated breadth25 encoder/decoder pair and calibrated source orientation.
All 23 actuators remain present, including both six-joint legs. No fallback,
robot transport or physical actuation. The 667-frame resampled source loses
only the existing 10-frame packet warmup and one tail interval; no speed change,
added hold or success-prefix cropping. Initialization is source-pose placement,
not normal-standing acquisition or firmware handback.

Source: [pinned public recording from TWIST2's PICO-based workflow](https://github.com/amazon-far/TWIST2/blob/d5c7108e9ef82d1b8770e5b692f27a1294f3aa8a/assets/example_motions/0807_yanjie_walk_002.pkl).
It is retargeted robot motion, **not raw headset packets or verified XR24 input**.

- Leg-joint RMSE: **0.207371 rad**; arms: **0.596487 rad**.
- Pelvis-relative foot-position p95: **0.266102 / 0.208334 m** (left/right).
- Root-position p95: **1.056420 m**. Root XY is not an explicit input in this
  causal 267-value interface, but foot/joint errors independently remain large.
- Recorder `passed: true` means complete stable integration only. Tracking and
  physical full-body VR remain **NOT READY**. No new controller improvement.

Video: `C:/Users/camer/sonic23_sim_artifacts/public_vr_legs_20260910_v1/walk002_paired/source29_vs_measured23.mp4`.
Left panel is prescribed original29 source, not a dynamically simulated teacher;
right is the measured native23 trajectory. Shared fixed-world camera, one
initial yaw/XY alignment, no fitted lag or per-frame recentering. All 657 saved
states rendered at 50 Hz; 13.14 s video includes the initial-state frame.

The same directory contains `report.json`, `tracking.json`, `render.json`,
`measured_trace.npz`, `command_diagnostic.json` and `observed_commands.npz`.
Read-only policy observation reproduces qpos/qvel/time **bit-exactly**. Reference
knees average 0.0825/0.1103 rad; policy targets 0.3336/0.3618 rad; measured knees
0.3716/0.4323 rad. This confirms sustained tracking bias, not disabled legs.
Target angles are balance-control requests, so target/reference mismatch alone
does not prove a decoding bug. Earlier buffered timing, target-codec and knee
envelope experiments already exist in PROGRESS.md; do not repeat them blindly.

Source trace SHA256:
`00f1695879d36229c3f02e623cfbb5c57e70ef360d55942166e9e39ba90c7a0a`.
Fresh replay matches the historical paired-orientation baseline exactly; this
is newly executed/visible evidence, **not a policy gain**.

## Current status

- Live PICO-to-MuJoCo diagnostic entry point exists:
  `gear_sonic.scripts.run_g1_true23_frozen_lora_live_teleop`.
- Use the validated breadth25 encoder/decoder pair in
  `artifacts/g1_true23_frozen_lora/paired_encoder_20260905_v2/original_breadth25/`.
  The older residual decoder/live qualification JSON is historical, not this pair.
- Fresh model-load preflight passes with 23 actuators and the paired model;
  no physics/control steps run. 33 VR receiver, tracking-health and XR24/SOMA
  regression tests pass (41.98 s). These are software checks, not a VR session.
- Receiver/input checks plus the existing VR regressions now pass 50 tests
  (39.71 s). Missing receiver, failed tracking probe, unavailable health,
  positive tracking health without hardware authorization, and LF startup
  are explicitly covered. This does not improve the existing policy's tracking.
- Receiver startup fault fixed: the Bash launcher had CRLF line endings.
  It now uses LF with an explicit Git checkout rule. XRoboToolkit receiver
  is running, listening on local TCP60061 and LAN TCP63901. The SDK now
  confirms its server connection. No robot controller was started.
- Latest input status is `receiver_reachable_tracking_unavailable`:
  no valid headset tracking-health packet or body stream. Zero-filled SDK
  defaults alone cannot diagnose headset power or calibration. Service and
  binding hashes match their pins. ADB reports no attached device.
  Wi-Fi IPv4 is **192.168.1.7** at this check.
- Physical full-body VR is **not ready**: live input, full-body tracking,
  standing acquisition and physical handback remain unqualified. Connecting
  the headset alone does not fix the observed turning/foot-tracking failures.

Evidence for this check:
`C:/Users/camer/sonic23_sim_artifacts/vr_priority_20260910_v1/pico_health.json`.
Model-load evidence: `sim_preflight.json` in the same directory.
Updated receiver/input evidence: `input_status_receiver_started.json` and
its separately hash-bound `.tracking.json` report in the same directory.

## Receiver startup and unambiguous input check

Run the receiver in WSL if it is not already running; do not start duplicates:

```bash
bash /mnt/z/codex/GR00T-WholeBodyControl-sonic-transfer-23dof/install_scripts/run_pico_robotics_service.sh
```

In another WSL terminal, use the new read-only precheck. Unlike the historical
tracking-only probe, it reports a missing receiver before opening the SDK:

```bash
cd /mnt/z/codex/GR00T-WholeBodyControl-sonic-transfer-23dof
PYTHONPATH=.:/mnt/z/codex/GR00T-WholeBodyControl/external_dependencies/XRoboToolkit-PC-Service-Pybind_X86_and_ARM64 \
  /usr/bin/python3 -m gear_sonic.scripts.probe_g1_true23_pico_input \
  --duration-seconds 3 --output /tmp/pico-input-new-session.json
```

Choose a fresh report path. `tracking_ready` means only tracking-health passed;
it never authorizes physical control. A local TCP check does not prove the
headset-to-PC LAN route. Existing firewall scopes were inspected, not changed.

## Separate remaining test — real VR, simulation only

1. Connect XRoboToolkit on PICO to the PC, pair and calibrate the body trackers.
   The existing full-body path requires two unique trackers and all 24 body roles.
2. Verify fresh tracking health, then run the existing same-WSL producer/consumer
   with the paired model. Start with neutral standing, small arm motion and
   weight shifts; then controlled squat, step and turn. Stop on a simulator fault.
3. Measure hands/head/feet and lower-body tracking, source-to-control latency,
   rate/range/effort compliance and uninterrupted balance. Test pause,
   disconnect and reacquisition explicitly. A transport PASS is not motion PASS.
4. Qualify standing acquisition/return before any separately supervised physical
   handoff trial. No dance performance is required for this VR acceptance path.

The current diagnostic initializes MuJoCo at the first source pose; that is
**not** a standing-acquisition test. Existing fallback tests also do not prove
physical Unitree ownership handback. Keep these gaps visible in every report.

Existing simulator commands are in
`docs/source/getting_started/g1_true23_frozen_lora_live_teleop.md`, under
"Live PICO to MuJoCo". Choose fresh output paths and verify the installed PICO APK
hash; do not copy a guessed hash or use historical physical-launch commands.
