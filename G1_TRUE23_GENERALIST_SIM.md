# SONIC native23 generalist — simulation implementation

Status: training and evaluation infrastructure runs; general dance/teleop policy
is **not qualified**. No robot, DDS, mode switching or hardware limits changed.
Original `23dofsonic` repository and dirty hardware work remain separate.

Existing BONES-SEED data has now been located and verified in WSL; dataset
access is not the current blocker. See the corpus section below. The first
evaluated continuation to root200 survives the full timeline but regresses
tracking and final standing, so it is rejected for promotion. A second,
explicit measured-posture objective also regresses dance tracking: root p95 is
0.884/1.047/1.336 m. Lower final speed is not a passing result. See
`artifacts/g1_true23_generalist/root_feedback_posture_20260907_v1/comparison.json`.

Latest milestone: full planned happy-dance reference now passes native23
kinematic/support gates (546 source frames, 1,091 adapted frames, 2x duration,
10.04% adaptation). A separately versioned root-feedback actor and fixed-world
training task are implemented. Neither milestone qualifies the live controller.

Current correction: ordinary standing return now targets the **planned terminal
XY/heading**, not the initial world origin. Legacy return generated an8m/2s
translation while blending standing, so its whole-lifecycle8m error was mostly
a bad return request. Endpoint v2 preserves the full source/denominator. Parent
policy survives all1841 controls in three endpoint tests but still fails fidelity
and standing posture; first100-update candidate regresses and is rejected.

Second feedback-priority root100 completes all 1,841 controls in nominal,
X-push and Y-push CPU runs, including all 1,091 adapted dance samples. Nominal
source landmark p95 errors improve 7.6–20.5% over the preserved parent, but
X-push errors worsen and final standing joint error rises from 0.645 to
0.681 rad. Root position p95 is 0.655/0.969/0.471 m across the three cases.
It is still experimental, not promoted or deployment-ready. See
`artifacts/g1_true23_generalist/root_feedback_regression_20260907_v2/comparison.json`
for matched-input comparisons and all 104 verified training source bindings.
The previous root-feedback checkpoint passed 448 scoped tests. Dataset and
continuation additions subsequently passed 508 tests; the newer posture and
original-time audits passed their focused suites. Final combined results are
recorded in `PROGRESS.md`. These validate code, not physical dance readiness.

The [referenced SONIC-transfer method](https://sonic-agibot-x2.github.io/sonic-transfer/)
freezes the released platform and trains small decoder LoRA adapters around an
analytic embodiment codec. The approved native23 generalist plan here keeps
the encoder/token ABI but deliberately trains the **full native23 decoder**;
it is not a claim of exact replication or parity with that transfer. The
author also reports unresolved hardware torque-demand failures despite calm
simulation. Hence this implementation records requested effort separately
from saturated applied effort; simulator survival cannot authorize hardware.

## Architecture and important corrections

- Released low-latency SONIC encoder/FSQ frozen; 267 → 64 token ABI retained.
- All 18 native23 decoder weight/bias tensors train: 994 → 23, 37,390,871 parameters.
- Six missing physical axes stay absent. Canonical29 observation padding is a
  codec, not six virtual motors. No specialist policy switching.
- Exploration bounded to 0.02–0.5, initialized at 0.1; critic trained separately.
- Nominal native-model simulation uses configured gains/armature and full
  configured motor-effort saturation at 500 Hz; policy runs at 50 Hz. This is a
  simulator hypothesis, not verified manufacturer or hardware capability.
- Previous-action history is the transformed normalized requested target.
  Requested effort and saturation remain measured. Existing quarter-effort /
  target-slew gantry experiment is unchanged and is a different benchmark.
- Original passing walk evidence used different `released_retained` gains,
  reference-state initialization and loose fidelity gates. It never proved
  universal dance or standing-to-standing lifecycle success.
- The old dance retarget followed **recorded29 policy motion**, not original
  planned choreography. New planned-trace adapter consumes `planned_qpos50`
  only; recorded state cannot silently become the requested dance.

## Verified evidence so far

Actual RTX3070 smoke: four environments, two PPO updates, 64 transitions per
run. Every decoder tensor changed; all ten encoder tensors stayed bit-identical.
Exploration remained approximately 0.1. Both paired ONNX encoders are exact on
three parity probes. Decoder maximum absolute error is 2.473593e-6 for smoke v1,
and 2.115965e-6 for fresh smoke v2. Probe parity is not rollout qualification.

The final combined new test suite passes **315 tests, zero failures or skips**
in 135.39 seconds. Ruff E/F and format checks pass for all 37 new Python files.
The earlier 274-test receipt is retained as a historical snapshot.
The [final verification receipt](artifacts/g1_true23_generalist/verification_20260907_v2/verification.json)
binds all 37 new Python sources, 66 evidence files and unchanged active hardware
source/header hashes. SHA256:
`d7159d0c482b2c8249853a13f13d79b08296372767c9b96d7351bc79e19c0896`.

Shared CPU MuJoCo comparison, native-model gains, same references and starts:

| Policy | Reference-start controls | Standing/dance/return controls | Fidelity |
|---|---:|---:|---|
| Original walk v14 | 207 / 535 | 352 / 1296 | Fail |
| Previous lifecycle LoRA100 | 535 / 535 | 1296 / 1296 | Fail |
| New generalist **v1**, two-update smoke | 535 / 535 | 1296 / 1296 | Fail |
| New generalist **v2**, two-update smoke | Not rerun | 1296 / 1296 | Fail |

These counts mean integrated duration, **not successful dancing**. Lifecycle
includes all 546 source frames, plus standing/acquisition/return/proof. No
fallback, mid-run pose reset, history reset or alternate standing controller.
Smoke **v1** settles upright, final root speed about 0.0046 m/s, but its final
position is about [7.91, 1.39] m and elbows miss the requested standing pose.
Lifecycle task-point world p95 errors remain about 1.60–1.68 m.
These v1 results bind checkpoint SHA256
`990d7f2ee016cdec7e1f9be813d2fadda89fa43e89d4a625192971b6b72c597d`.
The v2 decoder tensors differ by up to 7.118913345e-6; v1 dynamics evidence
must not be reused as exact v2 evidence.

The separate v2 lifecycle rerun also integrates all 546 original source frames
and 1296 lifecycle controls. Source task-point world p95 remains 1.600–1.685 m;
final root is [7.90178, 1.39630, 0.75308] m, proof root speed peaks at 0.02705 m/s
and standing joint error reaches 0.6314 rad. V2 therefore also fails original
dance fidelity and standing location/posture. Its lifecycle report SHA256 is
`1e8f6dab199893a3a98ce4a6a8682a3d60edd36d535d5568ea3c461b8d71c587`.

Historical walk discrepancy is now isolated at the observation boundary:
the original controller reads angular velocity from stale, pre-final-substep
`cvel`, transformed by the current root orientation. The current controller
refreshes kinematics first. The first control's ten physics steps match
exactly; control 2's newest angular-velocity history then differs, changing
raw action by up to 0.127677 and the next requested torque by 4.23696 Nm.
The original controller reproduces the saved first 11 states bit-for-bit;
the current controller independently matches this benchmark's first ten
controls and 100 substeps bit-for-bit. Thus `historical_released_gains`
means matching gains, **not** exact historical observation semantics. This
does not diagnose the physical damping incident or justify stale live input.

Old adapted-reference vs original planned choreography world p95 discrepancy
is about 0.57–0.64 m. It instead closely matches recorded29 policy motion.
This reference defect must be fixed before treating training reward as dance
fidelity. Current benchmark's ankle-origin metric is not sole/contact evidence;
the final acceptance gate explicitly requires sole-contact tracking.

Evidence under `artifacts/g1_true23_generalist/`:

- `smoke_20260908_v1/update_verification.json`
- `smoke_20260908_v1/export/generalist.diagnostic.json`
- `smoke_20260907_v2/update_verification.json`
- `smoke_20260907_v2/export/generalist.diagnostic.json`
- `smoke_20260907_v2/v1_v2_tensor_comparison.json`
- `baseline_nominal_20260908_v2/report.json` and measured-state MP4s
- `lifecycle_nominal_20260908_v1/report.json`
- `lifecycle_nominal_20260907_v2/report.json`
- `first10_frontend_audit_20260907_v1/`
- `verification_20260907_v1/junit.xml`
- `verification_20260907_v2/junit.xml`
- `root_observability_20260907_v1/receipt.json` and its reproducible script
- `smoke_20260907_v3/` stage-transfer, update and source-binding receipts

Artifact directory date strings are run identifiers. Smoke v1 is a historical
source snapshot, not current-tree equivalence: later review added complete
Python dependency binding. Fresh smoke v2 uses that correction.

## Root-feedback architecture for source-world tracking

The legacy fixed interface has an observable limitation, not merely too few
training updates. With the reference held fixed, translating the robot by
[8, -3, 0] m leaves actual v2 encoder267, proprioception930, decoder994 and
raw23 action **bit-identical**. Changing root linear velocity also leaves the
instantaneous interfaces identical; velocity can only be inferred indirectly
from proprioceptive history and contact response. The actual training encoder
functions share this translation invariance.

The 267 channels contain reference leg history (240), reference-pelvis-local
hands/head positions (9), local orientations (12) and pelvis-orientation
conditioning (6). Proprioception contains angular velocity, joints, previous
actions and gravity, not root XY. The training body-position target also
reanchors XY to the robot. This can support root-relative choreography, but
the actor cannot recognize a persistent source-world XY offset as an error.
More training may improve gait and relative tracking; it cannot add missing
absolute-position feedback. Locomotion back to an initial origin would require
a separate feasible locomotion plan, not an ordinary posture-return blend.

The user's continuation directive approves the proposed separate root-feedback
change. `native23_root_feedback_actor` retains frozen SONIC267/token64 and adds
root feedback9 alongside decoder input994. Features are desired-minus-measured
root XYZ, desired linear velocity XYZ and measured linear velocity XYZ, all in
the current measured pelvis-yaw frame. Current desired position is received
q10; desired velocity is (q10-q9)/0.02 s. No future q11 is read.

A zero-initialized 9x4096 projection adds to the first decoder preactivation.
Initial means exactly match the old parent even for nonzero feedback; all18
decoder tensors plus 36,864 conditioning weights, bounded noise and critic train.
The new checkpoint and two-input ONNX contracts are distinct; old hardware
loaders must not load them. Physical world-pose/velocity estimation remains
unqualified. No world-frame acceptance thresholds change.

The new environment uses fixed-world q10 tracking objectives while preserving
q9 tokenizer properties. Rewards/terminations evaluate post-physics against the
held received reference before command advance. It explicitly records existing
MJLab 2 ms stale derived reward state; actor observations use refreshed state.

## Corpus and retargeting

Corpus audit binds original recording IDs, license/lineage evidence, named
joints, file hashes, timing and derivative parents. Deterministic family-wise
80/10/10 split occurs **before** mirrors, crops, tempo variants or retargets.
Every training span must exactly match an audited train-split asset across all
six motion arrays. A hash-valid manifest does not independently establish the
truth of license declarations or physical feasibility.

The initial 127-file repository inventory omitted the existing WSL BONES-SEED
download. The corrected local source inventory is:

- Archive `/root/bones_seed/g1.tar.gz`: 23,499,973,647 bytes, SHA256
  `52580ea8bced72ea9e2ff1e7b68f01c51c7f1099581e9a46b7c87e1dec106d8a`,
  matching the pinned release `2f59b2077b9da34dd4e43618e705c7cb962c9a66`.
- CSVs `/root/bones_seed/g1_extracted/g1/csv`; metadata
  `/root/bones_seed/metadata/seed_metadata_v004.csv`.
- 142,220 metadata rows group into 71,132 original capture takes; 98,854
  nonempty CSVs include 8,229 dance files, some mirrored. Empty/missing files
  are explicitly counted rather than treated as complete motions.
- Frozen full-metadata capture split has 550 test dance groups, 423 with
  nonempty original files. Neither source presence nor splitting qualifies them.
- Twelve train-split originals have complete lossless named29 conversions and
  independently verified archive membership. Native23 fitting and dynamics
  qualification still determine which references may enter controller training.

Entry points: `prepare_g1_true23_bones_seed` (`index`, `select-training`,
`convert-source`), `audit_g1_true23_bones_seed_archive`,
`retarget_g1_true23_bones_seed_source`, and
`audit_g1_true23_bones_seed_source_timestamps`. Outputs use exclusive creation and remain
under the Git-ignored `artifacts/g1_true23_generalist/bones_seed_local_*/` tree.
No raw data or metadata is pushed. Motion Data by
[Bones Studio](https://bones.studio/); use is subject to the
[BONES-SEED license](https://bones.studio/info/seed-license).

The three initial complete dance fits fail the unchanged protected-task gate.
A standing-transition fit passes all 741 control-grid samples, but an added
audit of all 890 original 120 Hz source samples detects two between-grid
right-foot orientation violations. Thus it is not qualified training data.
The audit recomputes FK with declared interpolation; it never treats stored
achieved task positions, clip survival, or a grid-only pass as full fidelity.
The original 5,318-clip joint-deletion corpus is not a substitute for this check.
Existing seven-request material plus synthetic standing remains smoke-only.

Offline adapter uses actual native23 MuJoCo task-space IK. Maximum adaptation
is 2× duration and measured 20% task-space excursion distortion. Original and
adapted errors, source-time map and contact schedule remain separate. Every
frame must satisfy constraints; rejected clips emit no accepted reference.
The target hand convention stays wrist-roll + 18 cm, matching current encoder
features. The neutral 8.4 cm source/target hand discrepancy is reported, not
hidden through a longer phantom hand proxy.

Input role matters: measured `robot_state` must satisfy source-model limits.
Explicit `requested_choreography` may contain unattainable29 joint requests;
original targets and limit excess remain recorded, while native23 output
constraints are unchanged. The full planned dance includes 70 such frames,
maximum source excess about 0.0454 rad.

The full-source v3 sweep solves all 12 candidate trajectories: 546, 682, 818
or 1091 frames, preserving the original 546-frame timeline through an explicit
time map. None passes the unchanged final constraints. When the legacy
whole-horizon lower/root seed introduces invalid frames, the new narrow
fallback tries all23-joint fixed-root IK; it does not accept the failed seed.
At 2× duration and 90% excursion, foot p95 errors improve below 1 mm, but
hand p95 errors remain 17.49/18.74 cm and head 15.13 cm. This is a measured
fixed-root solver failure, not proof the choreography is physically impossible.
No accepted reference is emitted; rejected attempts remain in the report.

The follow-up v4 uses one 2×/90% candidate and constrained root-orientation +
23-joint refinement. This addresses a structural issue: native23 head-position
Jacobian with respect to its 23 joints is zero when root is fixed. Root motion
is a reference variable, not an extra actuator. All 1091 output frames complete;
head p95 improves from 15.13 to 3.21 cm, hands from 17.49/18.74 to 5.84/6.12 cm,
and foot p95 stays below 0.75 mm. Joint speed/acceleration and serialized root
bounds pass. Distortion is 10.04%, root translation reaches 3.804 cm and root
rotation reaches the 0.45-rad L1 bound.

V4 still **rejects**: only 252/1091 frames pass the unchanged protected mask.
Mean foot-orientation error increases; the aggregate report cannot attribute
all 839 failures solely to orientation versus COM/other protected criteria.
The root optimizer is weighted, not lexicographic. Final protected gates cover
feet, COM and total weighted error, without an extra head-orientation guarantee.
Nineteen refinement steps are accepted before nonlinear stalling; this is not
a certificate of physical infeasibility. No further sweep or threshold
relaxation is performed. Next solver change would need explicit foot-orientation
and COM constraints before upper-body fitting, not another weighted fit.
Full evidence: `planned_dance_retarget_20260907_v4/report.json`, SHA256
`0f32fe59e30001945479a10e9d61e996577fb1bab576a4b839131877082d3657`.

V4 forensic replay now identifies 839 unique invalid frames: overlapping COM
regression170, left-foot orientation267 and right-foot orientation512. The
rejected path is saved in diagnostic-only format, not a training-motion NPZ.

V5 adds explicit hard second-order-cone constraints for both foot positions and
orientations, COM regression and each frame's original weighted-cost ceiling.
Five iterations produce an accepted full path: all1091 protected frames pass,
head p95 3.76 cm, hands6.85/7.12 cm, foot maxima1.81/2.62 mm. Native ROM, raw
action, velocity/acceleration and serialized root bounds pass unchanged. Duration
is2x and task-space adaptation10.04%. Cold-file FK and raw planned-source audits
independently pass. See `planned_dance_retarget_20260907_v5/report.json`,
`saved_motion_verification.json` and `source_lineage_verification.json`.
This establishes a feasible reference under the stated kinematic tests, not
actuator/contact-force feasibility, dynamic tracking or arbitrary-dance coverage.

Causal reference core shares per-frame IK tasks, never reads future poses and
requires explicit contacts, current reference initialization and continuous
50 Hz timestamps. Rejection emits no replacement pose and requires explicit
reinitialization. This is reference generation, not motor recovery. Real-model
100-frame standing and small-shoulder tests each accept 100/100, but measured
moving p95 is 24.40 ms with 40/100 missed 20 ms deadlines: **not realtime-ready**.
Root-path optimization and PICO packet integration remain outstanding.

## Entry points

Use the pinned WSL Python `/root/.venvs/g1_true23_mjlab/bin/python`, from the
transfer checkout. Set `OPENBLAS_NUM_THREADS=1`, `OMP_NUM_THREADS=1`,
`MKL_NUM_THREADS=1` and `PYTHONPATH` to this checkout plus the original
`external_dependencies/unitree_rl_mjlab`. Nothing here uses DDS.

```bash
python -m gear_sonic.scripts.train_g1_true23_generalist smoke \
  --source-checkpoint low_latency/last.pt \
  --warm-start sonic_release/g1_23dof_rev_1_0_low_latency_init.pt \
  --motion-file artifacts/g1_true23_frozen_lora/standing_motion_ppo_20260906_v1/corpus/corpus.npz \
  --motion-metadata artifacts/g1_true23_frozen_lora/standing_motion_ppo_20260906_v1/corpus/corpus.recovery.json \
  --spans artifacts/g1_true23_frozen_lora/standing_motion_ppo_20260906_v1/corpus/corpus.spans.json \
  --run-dir artifacts/g1_true23_generalist/NEW_EXCLUSIVE_RUN \
  --num-envs 4 --iterations 2 --save-interval 2
```

`train` requires `--corpus-manifest`; unreviewed local regression inputs cannot
be promoted by changing the command name. Sessions are bounded to 100 updates
before independent CPU evaluation. The base launcher above uses nominal,
clip-contained reference acquisition. The separate
`train_g1_true23_generalist_curriculum` launcher now implements nominal
`--curriculum-stage acquisition` and `--curriculum-stage lifecycle` modes.

Each stage validates the original train-split inputs first, then writes derived
references to a new `--curriculum-directory` separate from `--run-dir`.
Acquisition explicitly holds only the first source pose; lifecycle retains
every source frame between standing/acquisition/return phases. Clip sampling
and inherited robot-state writes are permitted only during environment reset.
The new command bypasses timed resampling, clamps the final reference and lets
the episode timeout normally. Generated ramps are not contact/force-qualified.

Use `--initialize-actor-from <native23_generalist_model_N.pt>` for an explicit
stage transition: preserve the validated actor and bounded noise, initialize a
fresh critic/optimizer/counters and bind parent checkpoint plus new lineage.
This differs from same-stage `--resume`, which requires identical derivation,
parent and lineage. Audited production transfers require the same corpus and
recording-family split; an unaudited smoke parent cannot silently enter them.

The v3 lifecycle smoke transfers v2 actor tensors **29/29 exactly**, then changes
all 18 decoder tensors while preserving all ten encoder tensors. All 86 bound
local Python sources still match the run snapshot. Four environments execute
16 controls each: 64 transitions, two updates, four initial resets, **zero**
completed reference timelines. Only initial standing is exercised. This tests
stage plumbing, not motion acquisition, complete lifecycle tracking or general
dance performance. No v3 rollout/ONNX qualification is claimed. Randomized,
delayed/interrupted teleop curricula and the broad training campaign remain
unimplemented/unrun; do not call the overall curriculum complete.

Other CLI modules provide `--help`:

- `audit_g1_true23_generalist_corpus`
- `retarget_g1_true23_generalist_offline`
- `retarget_g1_true23_generalist_planned_trace`
- `export_g1_true23_generalist`
- `verify_g1_true23_generalist_update`
- `evaluate_g1_true23_generalist_baselines`
- `evaluate_g1_true23_generalist_lifecycle`
- `train_g1_true23_generalist_curriculum`
- `train_g1_true23_root_feedback`
- `export_g1_true23_root_feedback`
- `verify_g1_true23_root_feedback_update`
- `evaluate_g1_true23_root_feedback`

The root-feedback launcher adds explicit `regression` mode for one bounded
local experiment (<=100 updates, <=32 environments). It does not make local
clips an audited corpus. `train` still requires an ownership/split manifest.
Use separate new curriculum and training directories; existing outputs are
never overwritten. Root checkpoint names are `root_feedback_model_N.pt`.
The initial old-generalist actor can be transferred with
`--initialize-actor-from`; root exact resume requires matching lineage.

New root runs use `--return-target planned_endpoint` and
`--optimizer-profile feedback_priority`: base/decoder rate5e-7, conditioner
200x and critic600x that base, exploration1x. Profiles and all effective rates
are lineage-bound; fixed PPO schedule cannot silently flatten the rates.
Historical reproduction requires explicit `--return-target configured_origin`
and `--optimizer-profile legacy_uniform --learning-rate 5e-6`. First candidate's
same-state action audit measured decoder drift about470x its new feedback effect;
the differential-rate experiment addresses that imbalance, without claiming
overall controller improvement: dance p95 decoder drift falls to0.069 and
feedback effect rises to0.01623 on the same saved parent observations. Full
independent rollouts show mixed tracking results and failed standing return.

Root paired export is `obs_dict[1,994]` plus `root_feedback[1,9]` to
`action[1,23]`, with the unchanged frozen encoder267/token64 in its own file.
Its schema2 diagnostic manifest cannot authorize hardware. CPU evaluation uses
one fixed policy across full standing/acquisition/dance/return phases, with
nominal and two explicitly scheduled force cases. No pose writes after reset,
fallback controller or reference shortening is introduced.

Use original repo `--asset-root` for the CPU referee: its hard-pinned XML has
LF bytes; transfer XML is CRLF-equivalent but correctly fails the original
byte-hash pin. Do not weaken the model pin to hide this difference.

## Remaining work before simulation completion

1. Train and qualify the new root-conditioned policy; implementing observability
   alone does not prove drift correction. Qualify its physical estimator later.
2. Expand accepted **planned** references beyond the one full dance now passing
   kinematic gates; verify dynamic tracking and contact-force feasibility.
3. Finish source-to-feasible-native23 ingestion from the existing BONES-SEED
   archive. Capture-group splits and selected source-byte provenance are now
   frozen; corpus manifest integration and accepted broad references remain.
4. Train full decoder through acquisition, diverse motion, full lifecycle and
   randomized/delayed/interrupted input curricula. The two single-reference
   100-update local regressions are not that broad campaign.
5. Close causal retarget timing and wire PICO reference packets through the same
   controller; validate saved and paced input, stale input and standing return.
6. Qualify one checkpoint on at least100 held-out dances, all three perturbation
   seeds, >=95% complete lifecycle success; source distortion separate. Rejects
   and missing cases stay in denominator. Bind immutable full-source phase
   plans so self-declared shortened clips cannot pass. Diagnose after three
   unchanged 100-update evaluations; never silently relax the thresholds.
7. Complete sole/contact, no-fall, actuator, ONNX and full-video evidence.

Hardware handoff, physical motor limits and operator-supervised live testing
are a later, separate qualification. Simulation completion never authorizes
automatic hardware execution.
