# Bounded native23 shoulder-recovery experiment

This simulator-only diagnostic investigates whether one legal right-shoulder-roll target can survive a 32 ms coupled-robot preview across application delays of 0–12 ms. It does not change the production strict runner, actor, gains, limits, rewards, or hardware behavior.

## Executed result

Completed study: `E:/codex-artifacts/native23_shoulder_recovery_20260914/RESULTS.md`.

Exact replay: 31,696 physical steps. At rejection, 0/318 sampled targets passed all seven delays; 20 ms earlier, 314/317 passed. Full controller restoration matched the uninterrupted continuation exactly. Three earlier interventions lasted 63.50, 79.72 and 63.54 seconds against a fresh baseline of 63.38 seconds. All ultimately met strict rejection at the same right-shoulder lower boundary. None completed the motion, passed tracking or reached terminal standing. Timing remains unqualified.

No candidate promoted. Study stopped as bounded. Evidence supports investigating earlier, sustained state-based braking, but does not demonstrate that such a runtime rule will succeed.

## Reproduce

Run in the existing WSL Ubuntu-22.04 runtime, with E: mounted at `/mnt/e`. Use a new output directory for each execution; the scripts reject existing output directories.

```bash
PY=/mnt/e/codex_sonic_runtime/mjbatch323_20260910/venv/bin/python
SRC=/mnt/z/codex/GR00T-WholeBodyControl-sonic-transfer-23dof/artifacts/onboard_inspection_20260912
INPUT=/mnt/e/codex-artifacts/native23_preview_repair_20260913/strict_newer_v1
OUT=/mnt/e/codex-artifacts/native23_shoulder_recovery_new_run

"$PY" "$SRC/diagnose_pico_preview_failure.py" \
  --input-run "$INPUT" \
  --snapshot-controls 3169 3168 3167 3164 3159 \
  --output "$OUT/replay"

"$PY" "$SRC/search_pico_shoulder_recovery.py" \
  --input-run "$INPUT" --reconstruction "$OUT/replay" \
  --output "$OUT/search"

"$PY" "$SRC/continue_pico_shoulder_recovery.py" \
  --input-run "$INPUT" --search "$OUT/search" \
  --output "$OUT/continuation"
```

Required existing assets are the actor and factory configuration named in the input run's manifest, the native23 model/contract bundle, Pico references, and the pinned native preview library. Replay checks the manifest's physical model and binary hashes. This experiment used MuJoCo 3.2.3 on x86_64 Linux; equality claims apply to the verified environment.

## What each stage establishes

1. Replay compares positions, velocities, and commanded torques at every available 2 ms step. A mismatch reports the first component and prevents snapshot release. Snapshots contain `mjSTATE_INTEGRATION`, including warm-start state. Restored serialization is checked against the original continuous plant.
2. Search uses 257 uniformly spaced legal targets, six additional seeds before deduplication, and two bounded refinement rounds. All other 22 requested targets stay fixed. Every candidate runs at all seven delays in the full coupled model with the original PD law and per-step torque saturation. CSV files show each candidate/delay; NPZ files retain requested and saturated effort separately. Full JSON reports contain per-joint margins. The native guard's cold-state fixed-candidate verdict is separate from the warm-state physical result.
3. Continuation starts a fresh uninterrupted controller run and captures snapshots after packet admission, before importing current observation history or generating a command. It verifies restored ONNX features, targets, every subsequent integration state, commanded torque, and rejection against the uninterrupted tail. Only then does it try up to three distinct earlier candidates: best worst-case margin, nearest positive-effort saturation seed, and greatest legal surviving target. Each is checked on the fresh branch state at all seven delays first. Later targets come from the actual controller responding to the changed state. Strict rejection remains active.

## Continuation interpretation

This is an offline logical-time experiment. It reuses archived admitted sequence indices and archived command-delay values where available, then uses causal 50 Hz packets and a declared 12 ms delay. New logical admission times are at control boundaries. It is **new evidence**, not an assertion that the original asynchronous controller trace was reproduced.

The plant advances ten 2 ms steps per control. Computation pauses simulation time, so these runs cannot pass an independent-clock deadline gate. Tracking, physical completion, terminal 30-second standing, and timing are reported separately. Timestamp-selected intervention is a counterfactual, not a deployable trigger.

A sampled short-horizon survivor is not a demonstrated recovery. Sampling failure is not an impossibility proof. The study stops after the bounded comparison; it does not automatically start training, expand to whole-body optimization, or weaken the strict stop.

## Focused checks

```powershell
python -m pytest gear_sonic/tests/test_pico_shoulder_recovery.py gear_sonic/tests/test_g1_true23_sim_preview.py -q
```

The checks cover named joint/contract alignment, saturation seeds, unchanged delay-prefix causality, raw versus saturated effort, another joint disqualifying a candidate, warm-start restoration, divergence preventing snapshot publication, and the existing strict rejection behavior.

The continuation tool also accepts `--verify-existing PATH` to repeat the same saved local candidates, compare every physical step and retain terminal guard diagnostics. It reads the trusted local `coherent_snapshots.local.pkl`; use only snapshots generated by this experiment. This mode does not expand candidate selection.

Executed artifacts: `E:/codex-artifacts/native23_shoulder_recovery_20260914/`.
