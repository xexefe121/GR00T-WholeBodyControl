# True23 G1 frozen-platform LoRA transfer

This path applies the SONIC cross-embodiment transfer recipe to the physical
23-DoF Unitree G1. It is simulator-only. It does not authorize DDS, LowCmd,
gantry, or untethered robot execution.

Sources:

- [SONIC cross-embodiment project](https://sonic-agibot-x2.github.io/sonic-transfer/)
- [Draft paper](https://sonic-agibot-x2.github.io/sonic-transfer/static/pdfs/paper.pdf)
- [Released GEAR-SONIC platform](https://nvlabs.github.io/GEAR-SONIC/)

## Why this path exists

The earlier true23 teleop runs trained a sliced 23-output decoder. That learned
several Pico motions, but later updates also changed broad decoder behavior and
forgot motions such as hand crawl, happy dance, and deep crouch.

This path keeps the released platform immutable:

1. The released 267-input teleop encoder is frozen.
2. The exact FSQ token bottleneck is frozen.
3. Every source dynamics-decoder weight and bias is frozen.
4. Zero-effect LoRA A/B tensors are added to every decoder linear layer.
5. A fixed analytic codec enforces padded-29 true23 absent slots and selects
   the 23 physical action rows from the source 29-action output.
6. Only LoRA tensors and a fresh critic enter the PPO optimizer. Action
   standard deviation is frozen.

At initialization, the merged policy must reproduce the hash-bound true23
action-subset checkpoint exactly. Training refuses a source/warm-start lineage
mismatch.

| Source profile | LoRA rank | Trainable actor parameters |
|---|---:|---:|
| Normal SONIC | 16 | 245,744 |
| Released low-latency SONIC | 8 | 253,944 |

The normal rank-16 count matches the draft method's reported 245.7k budget.
The current Pico causal path uses the wider low-latency decoder, so rank 8
keeps a similar budget.

## Analytic 29-to-23 codec

The target robot is not a new kinematic family. It is the Unitree G1 with six
canonical joints absent. Existing true23 observations already use the source
H10 padded-29 layout. The encoder codec therefore fixes all absent joint slots
to exactly zero; audit validation rejects nonzero slots. Decoder codec performs
this fixed row
selection:

```text
native23 -> source29:
0,1,2,3,4,11,12,6,7,15,16,9,10,19,20,13,14,21,22,17,18,23,24

absent source joints:
5,8,25,26,27,28
```

Codec has no learned parameters. It solves representation alignment, not
dynamics mismatch.

## Two-phase training

Breadth uses the full retargeted multi-motion corpus. Sampling follows the
existing failure-driven adaptive bins, then clamps each sample inside its clip
with the causal H10 lead-in and proof-frame margins.

Polish uses near-uniform sampling over the full corpus. Exactly 10% of resets
come from a small curated target-behavior bank. Bank file must be bound to the
corpus span sidecar, feasibility-screened, override/deduplicated, and must not
replace the main corpus.

Example bank file:

```json
{
  "kind": "g1_true23_frozen_lora_behavior_bank_v1",
  "span_sidecar_sha256": "<sha256 of corpus.spans.json>",
  "feasibility_screen_passed": true,
  "override_and_dedup_complete": true,
  "clip_indices": [3, 19, 27]
}
```

## Breadth commands

Use a new run directory in this worktree. Supply the same corpus, metadata,
and spans already used by teleop-v14.

```bash
python -m gear_sonic.scripts.train_g1_23dof_mjlab_frozen_lora \
  preflight \
  --phase breadth \
  --source-checkpoint low_latency/last.pt \
  --lora-rank 8 \
  --lora-alpha 8 \
  --warm-start sonic_release/g1_23dof_rev_1_0_low_latency_init.pt \
  --motion-file /root/path/corpus.npz \
  --motion-metadata /root/path/corpus.json \
  --spans /root/path/corpus.spans.json
```

Replace `preflight` with `smoke`, then `train`. Recommended first production
settings:

```bash
python -m gear_sonic.scripts.train_g1_23dof_mjlab_frozen_lora \
  train \
  --phase breadth \
  --source-checkpoint low_latency/last.pt \
  --lora-rank 8 --lora-alpha 8 \
  --learning-rate 5e-6 \
  --warm-start sonic_release/g1_23dof_rev_1_0_low_latency_init.pt \
  --motion-file /root/path/corpus.npz \
  --motion-metadata /root/path/corpus.json \
  --spans /root/path/corpus.spans.json \
  --run-dir /root/g1_true23_runs/frozen_lora_breadth \
  --num-envs 128 --iterations 10001 --save-interval 500
```

Checkpoints use `frozen_lora_model_N.pt`. They contain adapter, critic,
optimizer, counters, immutable lineage, and merged-policy hash. They are not
deployment artifacts.

Materialize merged weights before a simulator evaluator consumes a checkpoint:

```bash
python -m gear_sonic.scripts.materialize_g1_true23_frozen_lora_diagnostic \
  --checkpoint /root/g1_true23_runs/frozen_lora_breadth/checkpoints/frozen_lora_model_500.pt \
  --warm-start sonic_release/g1_23dof_rev_1_0_low_latency_init.pt \
  --source-checkpoint low_latency/last.pt \
  --output /root/eval/frozen_lora_model_500.diagnostic.pt
```

Materializer reconstructs frozen source, applies adapter, checks merged hash,
and writes weights-only diagnostic schema. Output is accepted for custom
simulator evaluation, not promotion or robot deployment.

Export its decoder for MuJoCo/library-motion comparison. The exporter runs
static-shape ONNX checking and CPU ONNX Runtime parity, and writes no promotion
or deployment artifact:

```bash
python -m gear_sonic.scripts.export_g1_true23_frozen_lora_diagnostic_decoder \
  --diagnostic-policy /root/eval/frozen_lora_model_500.diagnostic.pt \
  --output /root/eval/frozen_lora_model_500.diagnostic.decoder.onnx \
  --report /root/eval/frozen_lora_model_500.diagnostic.decoder.json
```

## Independent gates and phase switch

Evaluate every saved checkpoint on four disjoint suites:

- in-distribution corpus;
- difficult tail motions;
- held-out OOD motions;
- survival in an independent second physics engine.

Do not select from training reward. Record one JSON result per checkpoint:

```json
{
  "checkpoint": "breadth/frozen_lora_model_500.pt",
  "update_count": 500,
  "phase": "breadth",
  "in_distribution": {"success_rate": 0.82, "mean_tracking_error": 0.18},
  "tail": {"success_rate": 0.68, "mean_tracking_error": 0.27},
  "out_of_distribution": {"success_rate": 0.63, "mean_tracking_error": 0.31},
  "second_referee": {"survival_rate": 0.84}
}
```

Append it to the immutable evaluation ledger:

```bash
python -m gear_sonic.scripts.record_g1_true23_frozen_lora_gate \
  --record /root/eval/model_500.json \
  --ledger /root/eval/frozen_lora_gate_ledger.json
```

Breadth moves to polish after three in-distribution gates fit inside a 0.002
success-rate plateau. Start polish from the ledger-selected breadth adapter in
a new run directory:

```bash
python -m gear_sonic.scripts.train_g1_23dof_mjlab_frozen_lora \
  train \
  --phase polish \
  --adapter-init /root/g1_true23_runs/frozen_lora_breadth/checkpoints/frozen_lora_model_1500.pt \
  --behavior-bank /root/path/behavior_bank.json \
  --source-checkpoint low_latency/last.pt \
  --lora-rank 8 --lora-alpha 8 \
  --warm-start sonic_release/g1_23dof_rev_1_0_low_latency_init.pt \
  --motion-file /root/path/corpus.npz \
  --motion-metadata /root/path/corpus.json \
  --spans /root/path/corpus.spans.json \
  --run-dir /root/g1_true23_runs/frozen_lora_polish
```

Adapter initialization imports only selected LoRA tensors. Critic, optimizer,
curriculum, and counters start fresh under polish lineage. Ordinary same-phase
continuation repeats the same `--adapter-init` provenance and adds `--resume`
for the polish checkpoint; it then restores full training state.

Stop after two consecutive OOD success declines. Keep peak OOD checkpoint,
even if in-distribution score or training reward continues improving. Passing
both simulator gates only makes checkpoint eligible for later promotion
evaluation; it never makes it deployment-ready.

## Original SONIC parity refinement

If a hash-bound original SONIC motion remains below parity, first rehearse it
through a new PPO phase and reject the phase when closed-loop survival falls.
For a narrow remaining gap, the diagnostic residual fitter can line-search a
bounded final-affine correction from the original true23 compatibility
rollout:

```bash
python -m gear_sonic.scripts.fit_g1_true23_frozen_lora_happy_residual \
  --repository-root /path/to/source-assets \
  --base-decoder-report /path/to/model_25.diagnostic.decoder.json \
  --output-dir /root/eval/happy_residual

python -m gear_sonic.scripts.evaluate_g1_true23_frozen_lora_happy_residuals \
  --repository-root /path/to/source-assets \
  --manifest /root/eval/happy_residual/manifest.json \
  --baseline-report /path/to/base/happy_dance.json \
  --output-dir /root/eval/happy_residual/closed_loop
```

Teacher-state RMSE never selects the scale. Closed-loop completion selects it,
followed by the full preservation suite and authentic saved-PICO replay. Use
`select_g1_true23_frozen_lora_happy_residual` only for a candidate that fully
passed the happy replay. The resulting ONNX/report remain simulator-only.

## Hard limitation

Decoder LoRA cannot recover information discarded by frozen encoder or FSQ.
The low-latency checkpoint was released with a specific reference semantic
profile. The Pico causal H10 signal is outside that original semantic contract.
This method reduces catastrophic forgetting and target-dynamics adaptation
cost; it does not prove that frozen tokens contain every causal behavior cue.
OOD and second-physics gates must expose that failure. If tokens are
insufficient, encoder-side adaptation needs a separately justified method.
