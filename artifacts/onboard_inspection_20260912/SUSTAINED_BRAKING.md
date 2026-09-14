# Persistent shoulder braking — bounded experiment

**Implemented and tested; rejected at startup. Leave disabled.** Both original-start attempts stopped after 20 ms. One correction was applied, then no sampled target met the declared terminal settling condition across all 49 delay pairs. Predicted physical limits remained satisfied. This is a failed filter design, not proof that braking or native23 teleoperation is impossible.

Results: `E:/codex-artifacts/native23_sustained_braking_20260914/RESULTS.md`.

## Frozen design

- Optional filter in `NativeTargetController.command`, before the existing final publisher. Default controller behavior remains unchanged when no filter is attached.
- TRACK requires a current command plus explicit braking continuation to pass. BRAKE recomputes a correction from current state every update. RELEASE requires three consecutive interior/nominal-continuation checks and individually tests each interpolated target.
- Only `right_shoulder_roll_joint` may change; the actual final target after native guard adjustment is checked again. A change to another joint is rejected by this experimental path.
- Fixed 80 ms horizon: one 20 ms current command, then three 20 ms feedback-braking controls. Other 22 current actor targets are held fixed in prediction; this assumption is not a prediction of future actor outputs.
- All 49 combinations of current and next delay, each 0–6 physics steps. The next delay repeats on the final two controls. This does not enumerate every longer jitter sequence.
- Backup captures shoulder position clamped 0.15 rad inside the joint range. It uses effective joint inertia and current model bias/passive forces to convert `400*(goal-q)-40*dq` desired acceleration to a legal target. Physical PD gains and torque saturation remain unchanged. Captured goal persists while BRAKE/RELEASE remains active.
- Terminal test requires shoulder clearance >=0.02 rad and absolute speed <=0.25 rad/s at both 60 and 80 ms. Release additionally requires current clearance >=0.12 rad and absolute speed <=0.1 rad/s for three updates. Release target step <=0.15 rad, with every candidate checked.
- Bounded candidates: feedback target, measured/actually applied targets, positive/negative saturation seeds, nine-point legal grid, and previous accepted correction. Feedback target is tested first; remaining unique seeds are ordered by distance from the actor request. No refinement or post-outcome tuning.
- Predictions start from measured q/v in fresh MuJoCo data; hypothetical branches do not access plant warmstarts, recording names, timestamps, future packets, or real controller memory. Snapshot reproduction separately preserves full plant integration state and warmstarts.
- Snapshots store filter mode, release counter, previous accepted correction, captured continuation and actual applied target. Wrapper configuration includes filter settings and native binary hash; disabled wrappers retain the earlier format.

This is a sampled simulator filter, not a formally guaranteed predictive safety filter. The cited [predictive safety filter paper](https://arxiv.org/abs/1812.05506) provides context; this experiment does not establish its safety assumptions or guarantees.

## Reproduce on the existing Linux runtime

Mount E: at `/mnt/e` in Ubuntu-22.04. Choose a new output directory for each attempt.

```bash
PY=/mnt/e/codex_sonic_runtime/mjbatch323_20260910/venv/bin/python
REPO=/mnt/z/codex/GR00T-WholeBodyControl-sonic-transfer-23dof
SRC="$REPO/artifacts/onboard_inspection_20260912"
OUT=/mnt/e/codex-artifacts/native23_sustained_braking_new_run
MJ="$($PY -c 'import mujoco; print(mujoco.__path__[0])')"
mkdir -p "$OUT"
gcc -O2 -std=c11 -fPIC -shared "$REPO/gear_sonic/utils/native23_sustained_prediction.c" \
  -I"$MJ/include" -L"$MJ" -Wl,-rpath,"$MJ" -l:libmujoco.so.3.2.3 -lm -o "$OUT/libbraking.so"

"$PY" "$SRC/check_sustained_prediction.py" --library "$OUT/libbraking.so" --output "$OUT/parity.json"
"$PY" "$SRC/run_sustained_braking.py" \
  --input-run /mnt/e/codex-artifacts/native23_preview_repair_20260913/strict_newer_v1 \
  --library "$OUT/libbraking.so" --output "$OUT/archived_delays"
"$PY" "$SRC/run_sustained_braking.py" \
  --input-run /mnt/e/codex-artifacts/native23_preview_repair_20260913/strict_newer_v1 \
  --library "$OUT/libbraking.so" --delay-schedule alternating_0_12ms --output "$OUT/alternating_delays"
```

The runner writes configuration/source hashes before the first command, all filter events and 49-pair predictions around interventions, physical trace, per-control mode/correction log, separate completion/tracking/standing/timing outcomes, and restored continuation comparisons. Reports exclude rejected commands from applied intervention duration.

## Validation and stop

Native/Python prediction parity: all 147 pairs across three archived observation states matched exactly. Focused tests: 32 passed, three MJLab-dependent tests skipped on Windows. TRACK and BRAKE controller/filter snapshots reproduced next commands and physical continuation exactly. RELEASE/hysteresis and label/offset independence passed focused tests; live RELEASE was never reached. Disabled controller path matched 30 saved baseline physics steps exactly, without repeating the closed replay study.

Both actual runs stopped before a second command could be applied. The prediction branches tested changing delays, but successful changing-delay closed-loop execution was not established. Full motion, tracking, terminal standing and real-time deadlines are unqualified. No timing optimization, horizon change, threshold relaxation, training, extra joints, or hardware run followed the failed design.
