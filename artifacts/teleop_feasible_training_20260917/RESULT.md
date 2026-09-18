VERDICT: KILL

# Feasible-reference training: result

Tested 2026-09-17. Hypothesis: training the existing frozen-LoRA recipe on a
corpus restricted to geometrically feasible references would improve tracking on
held-out walks. **It did not.** The new checkpoint is statistically
indistinguishable from the baseline, and the training scalars show why: under
this recipe, training episodes last about four control steps, so the policy
barely moves from its warm start regardless of the data.

## Decision rule, as pre-registered in PROGRESS.md

Pass only if, on held-out walk002, the new checkpoint completes all controls
without fallback **and** beats the baseline on both pelvis-relative hand p95
values **and** on all23 joint RMSE. Otherwise kill: do not extend, re-tune or
re-run the recipe.

## Result on the primary held-out clip, walk002

| Condition | Baseline | New | Outcome |
|---|---:|---:|---|
| Controls completed, fallback | 656/656, none | 656/656, none | met |
| all23 joint RMSE (rad) | 0.4192 | 0.4164 | better |
| Right hand p95, pelvis-relative (m) | 0.4493 | 0.4366 | better |
| **Left hand p95, pelvis-relative (m)** | **0.4649** | **0.4721** | **worse** |

The left hand is worse, so the rule is not met. **KILL.**

## All held-out clips

Neither policy was trained on walks 002, 003 or 008.

| Policy | Clip | Passed | Controls | legs12 | arms10 | all23 | L hand p95 | R hand p95 | Yaw p95 (rad) |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| baseline | walk002 | yes | 656/656 | 0.2286 | 0.5841 | 0.4192 | 0.4649 | 0.4493 | 2.249 |
| new | walk002 | yes | 656/656 | 0.2224 | 0.5823 | 0.4164 | 0.4721 | 0.4366 | 2.172 |
| baseline | walk003 | **no** (fell) | 595/596 | 0.2499 | 0.5724 | 0.4187 | 0.7498 | 0.6010 | 3.108 |
| new | walk003 | **no** (fell) | 589/590 | 0.2491 | 0.5736 | 0.4200 | 0.7533 | 0.6015 | 3.096 |
| baseline | walk008 | yes | 353/353 | 0.2805 | 0.5110 | 0.3935 | 0.7536 | 0.7629 | 2.534 |
| new | walk008 | yes | 353/353 | 0.2900 | 0.5085 | 0.3958 | 0.7375 | 0.7793 | 2.554 |

Every difference is under 3 percent and they point in both directions. On walk003
the new policy falls six controls earlier than the baseline. There is no
consistent improvement anywhere.

**Validity checks.** The baseline rerun on walk002 reproduced the numbers
measured on 2026-09-15 exactly, to every reported digit, so the evaluation
pipeline is deterministic. The 2026-09-17 imports of walks 002, 003 and 008 are
byte-identical to the 2026-09-15 imports (same `packet_sha256` and
`motion_sha256`).

The step-8 live-transport check was not run, because it was conditional on
passing step 7.

## Why: training episodes last four steps

Scalars from both runs' TensorBoard event files, first and last of 100
iterations:

| Scalar | Baseline run | New run |
|---|---|---|
| `Train/mean_episode_length` | 3.01 → 4.11 | 3.94 → 4.17 |
| `Train/mean_reward` | −99.85 → −99.77 | −98.93 → −98.73 |
| `Episode_Termination/stage_one_actuation_guard` | 8.19 → 7.50 | 7.63 → 7.25 |
| `Metrics/motion/error_body_pos` | 0.228 → 0.164 | 0.158 → 0.313 |

**The baseline had the same four-step episodes.** This is a property of the
recipe, not of the new corpus. Under `--actuation-profile native_support_stateful_v2`
the `stage_one_actuation_guard` terminates nearly every episode after roughly four
50 Hz control steps — about 80 milliseconds. Reward stays near −99 from the first
iteration to the last.

A policy that only ever experiences 80 ms of motion per episode cannot learn to
track a multi-second walk, whatever that walk's geometry. That is consistent with
every result here: two checkpoints trained on different data land within noise of
each other and of their shared warm start.

This is also a concrete candidate explanation for why earlier PPO campaigns on
this recipe did not improve tracking. It has not been proven to be the only
cause; it is the one visible directly in the training scalars.

## What this does and does not show

It shows that **reference feasibility is not the limiting factor for this
recipe**. Removing the infeasible PICO anchors and adding five feasible walks
produced no measurable change.

It does **not** show that feasible references are unimportant in general. A
recipe whose episodes run long enough to exercise tracking might respond to them.
Because training here barely moves the policy, this experiment could not have
detected a data effect of any size.

## Next lever, and what it is not

The next lever is **not** another training run on this recipe, with any data,
learning rate or iteration count. The four-step episode length would bound it the
same way.

It is a diagnosis: why does the actuation guard fire within 80 ms from the warm
start? Plausibly the warm-start policy's requested actions fall outside the native
23-DoF actuation support immediately, but that is a hypothesis to test by
inspecting guard trigger reasons at the first few steps, not something established
here.

## Exact commands

Corpus manifest, projection, build and training commands are recorded in
`PROGRESS.md` under Phase B1. Export and evaluation:

```bash
PYTHONPATH=<worktree>:/mnt/z/codex/GR00T-WholeBodyControl/external_dependencies/unitree_rl_mjlab
/root/.venvs/g1_true23_mjlab/bin/python -m gear_sonic.scripts.materialize_g1_true23_frozen_lora_diagnostic \
  --checkpoint .../main_run/checkpoints/frozen_lora_model_100.pt \
  --warm-start /mnt/z/codex/GR00T-WholeBodyControl/sonic_release/g1_23dof_rev_1_0_low_latency_init.pt \
  --source-checkpoint /mnt/z/codex/GR00T-WholeBodyControl/low_latency/last.pt \
  --output .../main_run/eval/model_100.diagnostic.pt
python -m gear_sonic.scripts.export_g1_true23_frozen_lora_diagnostic_{encoder,decoder} \
  --diagnostic-policy .../main_run/eval/model_100.diagnostic.pt \
  --output .../model_100.diagnostic.{encoder,decoder}.onnx \
  --report .../model_100.diagnostic.{encoder,decoder}.json
```

Held-out evaluation script: `/root/phaseC_simtest.sh`. Outputs:
`/mnt/e/codex-artifacts/teleop_feasible_training_20260917/simtest/` including
`summary.json`.
