# Frozen-platform LoRA versus the original true23 v14 trainer

> **Current status, 2026-09-06:** No native23 full-body dance, safe return or
> physical live-teleop qualification. The historical LoRA comparisons below
> used encoder `73335314...`, not the actual paired training encoder
> `3806b2b6...`; direct audit disagreed on all 535 happy-dance inputs. Those
> results cannot establish correctly paired SONIC fidelity or physical parity.
> Correct-pair retraining and controller-feasibility experiments still fail.
> No new matched-budget original-v14 comparison has been completed under the
> corrected pairing, constrained actuation and full-motion fidelity gates.

> **2026-09-05 correction:** The tables below measure historical completion
> under loose crawl-envelope gates, not dance fidelity or physical readiness.
> The selected 535/535 happy replay has 5.0104 m maximum pelvis error and
> 0.7170 rad maximum joint RMSE. Its controller gains/slew differ materially
> from the physical runtime; the deployed profile fails weight-bearing
> simulation. A separate full-clip `motion_fidelity` screen now accompanies
> new reports. See [current progress and evidence](../../../PROGRESS.md).
> No physical full-body dance/live-teleop readiness is established.

This comparison was run on 2026-08-18 in the isolated
`GR00T-WholeBodyControl-sonic-transfer-23dof` worktree. It compares the new
frozen-platform method with the original 23-DoF causal v14 implementation from
`GR00T-WholeBodyControl`. All results are simulator diagnostics. No DDS,
network, deployment, promotion, or robot command path was opened.

## Deployed-mechanics follow-up (2026-09-05)

The trainer now accepts opt-in `--actuation-profile stage_one_cpp`. It binds
the checked-out controller header and trains through explicit 500 Hz PD,
50 Hz inference, deployed gains, default-relative scaling and target slew.
The physical controller is not edited or authorized by this option. Legacy
training defaults remain unchanged. The older committed controller and the
uncommitted experimental header have different profiles; consult the saved
`stage_one_actuation` contract, not the option name alone.

A bounded 50-update run (16 environments, eight-clip SONIC/PICO corpus)
completed, but both update-25 and update-50 exports fail happy dance at 46/535
reference-start and 49/535 measured-start transitions with the experimental
0.60 fraction / 5 rad/s profile. Neither candidate is selected. This is a
diagnostic attempt, not a matched-budget retraining comparison against v14.

Measured-posture hold-only simulation also collapses with return-hold knee
kp=4. Training improvement alone cannot be assumed to repair acquisition or
handoff. See [current evidence and limitations](../../../PROGRESS.md).

## Correct-pair, constrained-controller follow-up (2026-09-06)

The V2 simulator profile seeds a synthetic controller state after the final
motion reset and feeds back the last applied target, not the requested target.
The new model completed 50 updates / 25,600 transitions; it was rejected.

| V2 experiment | Reference start | Measured state + 5 s standing | Full dance/return qualified |
|---|---:|---:|---|
| Original tempo, greedy effort projection | 50/535 | 16/535 | No |
| Original tempo, offline predictive projection | 50/535 | 72/535 | No |
| Half tempo, greedy effort projection | 45/1080 | 82/1080 | No |
| Half tempo, offline predictive projection | 49/1080 | 86/1080 | No |

The predictor uses a copied MuJoCo state and a bounded optimizer; it is not
trained into this model or qualified for real-time hardware control. Its
successful original-tempo interventions took up to 16.1 ms and its failed
searches up to 95.1 ms, exceeding the 2 ms physics/control period. It preserves
existing numeric limits and all 23 controlled joints, but does not prove
multi-step feasibility or safe recovery.

Half-tempo retiming retains all 546 original joint samples exactly, rebuilding
FK and velocities over 1,091 frames / 21.8 s. No joints, phases or root path
segments are removed. This is neither original-tempo parity nor a contact/COM
optimization; its first control frame is source phase 5 rather than 10, so it
is not a timing-only, same-initial-state ablation. The extra transition counts
must not be treated as proportional progress through the original dance.

Evidence is under `artifacts/g1_true23_frozen_lora/actuation_trace_20260905_v1`,
`predictive_projection_20260905_v1`, and `retiming_feasibility_20260906_v1`.
See [progress](../../../PROGRESS.md) for exact rejected-candidate hashes,
joint-level failure evidence and remaining hardware/headset boundaries.

## What changed

### Whole-body stance retargeting (2026-09-06)

New offline retargeting optimizes all 23 joints and root XYZ, rather than only
lifting the floating root. It preserves all eight source clips, all 6,035
samples and original 50 Hz timing. Explicit contact-family hypotheses, actual
mesh/capsule geometry, COM/foot/hand objectives and whole-path derivative
projection are recorded with per-frame solver and independent audit evidence.
The 5,955 complete causal history packets are rebuilt from new positions;
old proof hashes are not reusable. No deployment bundle or training corpus
was promoted, and no hardware controller, limit or interlock was changed.

Centering standing COM over ankle origins fixes a specific reference defect:
upright and standing anchors now clear both model floors and have conditional
in-limit support solutions on all 1,024 frames, in both static and pose-derived
inverse-dynamics screens. Grounding alone previously gave zero passing frames.
These screens retain optimistic near-contact/friction assumptions, and the
short PICO recordings retain their original terminal holds. This is not new
live input, closed-loop standing proof, or physical qualification.

Other clips remain rejected. Happy dance still penetrates the mesh/capsule
floor on 382/391 of 546 frames, with only 65 conditional in-limit reference
inverse-dynamics frames in either model. Maximum body-position change is
0.5007 m; 244 frame optimizations do not converge within the declared budget.
The soft collision objective and derivative projection are not a solved
contact-constrained trajectory optimization, nor original SONIC parity.

Using the unchanged, correctly paired model-50 and constrained V2 controller,
the new reference completes 54/535 reference-start transitions and 76/535 from
the historical recorded posture after 5 s standing. Previous floor-conditioned
counts were 50/535 and 16/535. Both new cases fail ankle target feasibility;
the requested standing return completes zero physics steps. Changed references,
remaining penetration and tracking errors prevent treating longer survival as
fidelity or readiness. No new matched-budget v14 training comparison exists.

The same paired controller also fails the improved upright/standing PICO
references at 50/1,013 and 44/1,013 transitions, respectively, with left ankle
target infeasibility and large knee-flexion deltas. Passing the reference
force-balance screen is not sufficient for learned closed-loop standing.

Reproducible artifacts and exact hashes are documented in
[PROGRESS.md](../../../PROGRESS.md), under `stance_retarget_20260906_v2`.
Live full-body teleop and exact reproduction of arbitrary 29-DoF motion on
23 physical axes remain unqualified. All 23 joints remain controlled; this is
not a substitution of an upper-body-only implementation.

### Reference support and PICO grounding (2026-09-06)

New offline support auditing keeps all eight clips and 6,035 frames. It checks
stationary poses separately from a finite-difference inverse-dynamics
hypothesis, using explicit ground-contact candidates and unchanged modeled
torque caps. It does not provide a controller, qualify dynamic tracking, or
establish physical motor ratings. Under that inverse-dynamics hypothesis,
only 75/546 floor-conditioned happy-dance frames have a conditional solution
within the bounds in either collision model. Unconditioned reference counts
are 89/546 and 88/546, but include penetrated geometry and are not superior
physical qualifications. This compares reference variants, **not v14 models**.

The PICO ankle-body height heuristic leaves standing anchors hovering roughly
13–14 mm over the mesh floor. Opt-in `--collision-grounding` now grounds the
actual foot geometry, retaining every 23-joint sample/proof and original
timing. All three saved PICO clips were rebuilt separately; legacy defaults
and the production training corpus remain unchanged. Their mesh hover is
fixed, but support/effort screens still fail and capsule-model overlap remains.
These are rejected teacher candidates, not teleop-ready clips. Full-body
contact/COM/lower-body retargeting is still required. The anchors include
their original terminal holds, not new extended live-headset recordings.
See [progress and exact limitations](../../../PROGRESS.md).

### Whole-reference floor conditioning (2026-09-06)

New offline conditioning clears every frame of the eight-clip, 6,035-frame
corpus against both training-capsule and evaluation-mesh collision models.
All 23 joint trajectories, their stored velocities, body orientations and
original timing remain unchanged; only bounded root-height translation and
its vertical-velocity correction are applied. No clip is removed. This is
geometric conditioning, not support/COM or contact-force optimization.

The same correctly paired model50 still fails happy dance at **50/535** from
reference start and **16/535** from measured-state simulation, with zero
successful return physics steps in the latter case. No retraining, model
selection or v14 matched-budget comparison occurred. These corrected-reference
tests therefore do not establish original SONIC fidelity, dynamic retargeting
success or live-teleop readiness. See [progress and artifacts](../../../PROGRESS.md).

### Exploration/reset audit (2026-09-06)

The std difference below was tested without changing trained weights: model50
with inherited action-noise scale 1.0, 0.25 and 0.0 completed episodes with
median length **3 control steps in every case**. The respective guard counts
were 921, 911 and 834 over 4,096 sampled transitions each. Sensor/observation
corruption remained enabled; later adaptive resets were unpaired. These are
training-distribution probes, not full-clip fidelity or a v14-weight rerun.
Reducing exploration alone is not a demonstrated fix and std remains unchanged.

An independent compiled-model contact audit also found floor overlap in 15/32
random initial states, including 5.1 cm and 2.9 cm overlap in two crouch states
that fail within their first policy interval. Geometric overlap alone does
not prove the causal impulse or explain every guard failure. Lifting only
penetrated synthetic reset roots removes the detected overlap and delays two
early crouch failures, but median episode length remains three control steps.
Unperturbed references still overlap the floor in 17/32 sampled states.
Combining perturbation removal and floor lifting still yields three-step median
episodes and 885 guard terminations over 4,096 sampled transitions. None of
these interventions changes production training, guards or model selection. See
[current progress](../../../PROGRESS.md) for the reset-intervention experiment,
exact hashes, numerical repeatability caveat and continuing hardware boundary.

| Boundary | Original true23 v14 | Frozen-platform LoRA |
|---|---|---|
| Actor initialization | Hash-pinned true23 recovery policy | Hash-pinned released low-latency 29-DoF SONIC platform plus analytic 29→23 codec |
| Frozen actor | Encoder and decoder through layer 12 | Encoder, FSQ, every base decoder weight/bias, and action std |
| Trainable actor | Full decoder layer 14 and output layer 16: 274,455 parameters in four tensors | Rank-8 LoRA at all nine decoder linears: 253,944 parameters in 18 tensors |
| Share of 37,393,949 decoder affine parameters | 0.734% | 0.679% |
| Exploration std | Frozen at 0.10 | Frozen released per-joint std (mean reported by RSL-RL: 0.38) |
| Sampling | v14 corpus sampler | Breadth adaptive sampler, then near-uniform polish with a fixed 10% screened behavior bank |
| Checkpoint selection | Training/evaluation campaign selected a late v14 candidate | Training reward excluded; peak OOD success, referee survival, tail/ID success, OOD error, then earliest update |
| Resume size | About 169 MB for the original full trainer checkpoint | 6,666,811 bytes for selected adapter/critic/optimizer resume |

The LoRA count is 7.47% smaller than the original v14 trainable actor count,
while distributing adaptation across the entire decoder. Its 6.67 MB resume
depends on the separately hash-pinned frozen source; the 163 MB materialized
diagnostic policy is deliberately a different artifact.

## Controlled run

Both breadth and original v14 use the same five-clip Pico corpus, seed
20260815, 64 environments, 16 steps per environment, and 100 PPO updates
(102,400 transitions). Frozen LoRA breadth was resumed exactly at update 25.
The breadth ledger plateaued at 0.80 in-distribution success and selected
`breadth/frozen_lora_model_25.pt` before polish.

Polish imported only that adapter and restarted critic, optimizer, curriculum,
and counters. Its fixed bank contains the two feasibility-proven walk clips
(indices 3 and 4); failed deep crouch was excluded. Polish ran through update
100. It never exceeded breadth-25 OOD success and regressed hand crawl after
update 25, so final selection correctly stayed on breadth-25.

The shared evaluation suite contains five in-distribution motions (two walks,
upright, standing, crouch), a difficult tail (the two walks and crouch), and
three OOD preservation motions (elbow crawl, hand crawl, happy dance). The
original model-100 decoder was rerun through this exact full-length suite to
avoid mixing its older 500-step anchor reports with the new evidence.

## Same-suite result

| Metric | Original v14 update 100 | Selected LoRA breadth update 25 | Change |
|---|---:|---:|---:|
| In-distribution success | 0.800 | 0.800 | unchanged |
| In-distribution mean max-relative tracking error | 0.4056 m | 0.3550 m | 12.5% lower |
| Tail success | 0.667 | 0.667 | unchanged |
| Tail mean max-relative tracking error | 0.4622 m | 0.4837 m | 4.6% higher |
| OOD success | 0.333 | 0.667 | +0.333 absolute |
| OOD mean max-relative tracking error | 0.6741 m | 0.4870 m | 27.8% lower |
| Completed/requested transition ratio | 0.7098 | 0.8325 | +0.1228 absolute |

Both pass walk001, walk010, upright, standing, and elbow crawl. LoRA also
passes all 595 hand-crawl transitions; original v14 fails after 194. Neither
passes deep crouch or happy dance. On the two remaining failures LoRA survives
103 versus 67 crouch transitions, and 449 versus 156 happy-dance transitions.

The selected artifacts are:

- resume: `artifacts/g1_true23_frozen_lora/pico_internet_breadth_100_seed20260815_canonical_v2/checkpoints/frozen_lora_model_25.pt`, SHA-256 `c79bea2669ec59d7ea560ebff358bbc3afe458370003d226b0135b63a6715c6f`;
- merged diagnostic policy: `.../eval/model_25.diagnostic.pt`, SHA-256 `033d5b3218506081d6390ba7f0b430e0e72b269b80e72cc51b9306d160926fae`;
- diagnostic decoder: `.../eval/model_25.diagnostic.decoder.onnx`, SHA-256 `c12038c5f606b59eda370779f9a877c1ecb27687d65c2310e9efc3af73ae1355`;
- combined gate ledger: `artifacts/g1_true23_frozen_lora/pico_internet_polish_seed20260815_canonical_v2/eval/combined_gate_ledger.json`.

The ONNX exporter validates static `[1,994] → [1,23]` float32 shape, opset 13,
full ONNX checking, shape inference, and three CPU ONNX Runtime parity cases.
Selected decoder maximum parity error was `2.145767e-06`.

## Original SONIC and saved-teleop parity follow-up

The follow-up binds original SONIC evidence by hashes instead of comparing
similarly named files. For happy dance, the chain is the released planner
motion, released 29-DoF policy rollout, released-policy true23 compatibility
rollout, its materialized 50 Hz true23 reference, and the candidate replay.
The released 29-DoF policy and true23 compatibility teacher both complete the
motion. Breadth-25 completes 449/535 native true23 evaluation transitions
(83.93%) before crossing the joint-tracking gate, leaving an initial 16.07%
completion-ratio gap to original SONIC.

An authentic saved PICO `walk001` causal packet clip was also replayed through
the selected LoRA decoder, not just through a motion-reference evaluator. It
completed all 684 transitions without fallback. Minimum base height was
0.7116 m and maximum tilt was 0.3071 rad against 0.30 m and 1.0 rad gates.
Live headset freshness and transport are deliberately not claimed by this
offline replay.

The parity polish corpus contains the original five Pico clips plus original
SONIC hand crawl, elbow crawl, and happy dance references. Near-uniform corpus
sampling preserves the successful behaviors; the fixed 10% screened bank
selects happy dance only. The phase imports only the breadth-25 adapter and
starts a fresh critic, optimizer, sampler state, and update counter.

The PPO rehearsal branches were screened and rejected. At updates 10 and 20,
the high-rate mixed run completed 390 and 246 happy-dance transitions; the
low-rate mixed run completed 432 and 377. A happy-only low-rate specialist
completed 386, 390, and 372 at updates 10, 20, and 30. All were below the
unchanged breadth-25 baseline of 449, so none replaced it.

The successful follow-up fits a bounded final-affine residual from the
hash-bound original SONIC true23 compatibility rollout, then uses closed-loop
MuJoCo—not teacher-state RMSE—to select its scale. Two scales completed happy
dance. Alpha 0.050 had lower happy-only error but regressed hand crawl to
250/595 and reduced eight-motion survival to 78.86%, so it was rejected.
The preservation-first alpha 0.002 candidate retained every previously passing
case and is the selected simulator diagnostic.

## Historical completion-only result (not current parity)

| Metric | Original true23 v14 | Breadth-25 LoRA | LoRA + 0.002 residual |
|---|---:|---:|---:|
| In-distribution success | 0.800 | 0.800 | 0.800 |
| Tail success | 0.667 | 0.667 | 0.667 |
| OOD success | 0.333 | 0.667 | 1.000 |
| Completed/requested transition ratio | 0.7098 | 0.8325 | 0.8465 |
| Happy dance | 156/535 | 449/535 | 535/535 |
| Hand crawl | 194/595 | 595/595 | 595/595 |
| Deep crouch | 67/1013 | 103/1013 | 100/1013 |

The original released 29-DoF SONIC policy and its true23 compatibility teacher
both complete 546/546 source control steps. The selected native true23
candidate completed its full 535/535 transition replay in the historical
experiment. Its completion-only ratio was 100%; this did **not** establish
motion fidelity, a correctly paired controller, or physical parity. Counts were
normalized because the source report includes its initial control step;
cross-embodiment joint errors are deliberately not compared.

The residual decoder also completed all 684 transitions from the authentic
saved PICO `walk001` causal packets with no fallback. Minimum base height was
0.6985 m and maximum tilt was 0.3298 rad. Thus the tiny residual can remain one
general simulator diagnostic instead of requiring a happy-only router.

Selected parity artifacts are:

- residual decoder: `artifacts/g1_true23_frozen_lora/original_sonic_happy_residual_v1/candidate.plus_0p002.decoder.onnx`, SHA-256 `44d1fb2701f1e65460f1c2c23f676bce4f1d4a44b3b112798dc5034af37946b8`;
- hash-bound decoder report: `.../candidate.plus_0p002.decoder.json`, SHA-256 `02197e5682a9bddc8f11aa6fa9c32ba909b97ec7d1c316c9a0d660cba2d25b7d`;
- parity summary: `.../parity_plus_0p002.json`, SHA-256 `ce289eb88a17ce8cd1f92e885d3297b800a62814da07e21c0c464b343e1ab06c`;
- preservation ledger: `.../candidate.plus_0p002.summary.json`.

## Live-transport parity and failover

The selected residual decoder now runs in a dedicated localhost live consumer
with its decoder, report, and candidate summary all pinned by SHA-256. The
authentic `walk001` packet clip was replayed through the exact ZMQ boundary at
50 Hz. Candidate and original true23 both completed 684/684 transitions with
no fallback. Candidate minimum height was 0.7126 m versus 0.6706 m original;
maximum tilt was 0.3399 rad versus 0.3648 rad original; maximum reference age
was 30.0 ms versus 24.77 ms original.

Unlike the original run, whose safety fallback was disabled, the selected
runtime keeps the balance fallback enabled. The prior fixed 0.25 rad tilt
trigger falsely interrupted the known-good walk envelope, so this profile uses
a 0.50 rad trigger while retaining the 1.0 rad hard physical gate. Timeout,
gap, and stale injection at live transition 120 each latched the corresponding
transport trigger and held stable balance for another 100 transitions.

See [Selected true23 frozen-LoRA live PICO test](g1_true23_frozen_lora_live_teleop.md)
for the real-headset commands and remaining hardware boundary.

## Physical gantry qualification state

On 2026-09-01, the selected residual decoder passed a fresh read-only G1
shadow on the physical 23-DoF robot: 100/100 accepted causal action frames,
mode-machine 4, no CRC rejection, and no position, slew, freshness, or
inference-deadline violation. The immutable evidence SHA-256 is
`c74d18e54c812912b2130d25352e5545b6a577b8985a6c5efe80ac52c5458aa6`.

The Windows-to-WSL replay boundary now samples the WSL clock once per packet,
uses a temporary 1 ms Windows timer period, and preserves exact 20 ms source
timestamps. The qualifying publisher's maximum schedule slip was 1.65 ms and
mean slip was 0.78 ms. The active subscriber uses a bounded queue rather than
conflation, waits up to 35 ms for the exact q9/q10 LowState bracket while
retaining the 40 ms state-freshness gate, and records packet age and inference
latency in terminal evidence. A causal repeat option can extend saved-clip
rehearsals without repeating or regressing source indices.

The physical active path passed artifact verification, stable advancing
LowState, motion-mode release, LowCmd publisher creation, and fresh-policy
readiness. It wrote damping commands only. No wireless A rising edge was
observed, so zero armed policy commands were written and no physical dance was
claimed. L2 deadman, A rising-edge arm, B/R2 stop, app/physical e-stop, joint
limits, target-rate limits, stale-policy damping, and the gantry-only promotion
remain mandatory. This is stronger than the original v14 implementation,
which had no promoted hardware command path, but physical motion parity is not
proven until one bounded armed gantry session completes its evidence contract.

## Limits

This is strong evidence for this eight-motion suite and one authentic saved
PICO packet clip, not a universal behavior claim. Deep crouch remains
unresolved. The frozen encoder/FSQ also cannot recover information it never
encoded. The referee here is the CPU MuJoCo reference implementation against
the MuJoCo-Warp training backend; it is an independent execution backend, not
a wholly different physics engine. Live localhost freshness and transport were
exercised, including watchdog faults. A currently connected real headset was
not available: the read-only health probe verified software hashes but found
no headset or trackers. Hardware and deployment remain unauthorized. A
broader held-out corpus, a truly separate physics engine, a real-headset
sustained session, and hardware safety qualification are still required before
any promotion decision.

Method source: [SONIC Transfer project](https://sonic-agibot-x2.github.io/sonic-transfer/)
and its linked paper.
