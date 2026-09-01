# Frozen-platform LoRA versus the original true23 v14 trainer

This comparison was run on 2026-08-18 in the isolated
`GR00T-WholeBodyControl-sonic-transfer-23dof` worktree. It compares the new
frozen-platform method with the original 23-DoF causal v14 implementation from
`GR00T-WholeBodyControl`. All results are simulator diagnostics. No DDS,
network, deployment, promotion, or robot command path was opened.

## What changed

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

## Parity-improved result

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
candidate completes its full 535/535 transition replay, so normalized
completion parity is 100% with a zero gap. Counts are normalized because the
source report includes its initial control step; cross-embodiment joint errors
are deliberately not compared.

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
