# G1 true23 SONIC — progress log

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

- Checkpoint SHA: `848f2bd69847198594278373c5e1f96557bbaf7b39b6947ef6088ed3393af3f8`.
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
