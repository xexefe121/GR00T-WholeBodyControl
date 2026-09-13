# Fixed-rate saved-input VR simulation: fault path passes, tracking still fails

This is a new simulation-only input-fault implementation and test, not a policy
upgrade or physical deployment approval. All inputs use the existing public
TWIST2 walk002 recording and the validated breadth25 encoder/decoder pair.
No new motion download, physics/limit change, training or robot command.

## Result

| Scenario | SONIC controls | Balance controls | First input fault tick |
| --- | ---: | ---: | ---: |
| Complete walking source | 656 | 0 | None |
| 500-ms pause at 4 s | 200 | 456 | 200 |
| Sequence gap at 4 s | 200 | 456 | 200 |
| Malformed packet at 4 s | 200 | 456 | 200 |
| Stale startup input | 0 | 656 | 0 |
| Complete source, then input ends | 656 | 250 | 656 |

All six requested timelines completed: 4,186 controls / 41,860 actual 2-ms
MuJoCo substeps. Every substep is retained, including the balance interval.
Independent recount in `verification.json` finds zero measured joint-range
excess, commanded effort at or below the unchanged caps, and maximum measured
joint speed at 0.5693 of the configured motor limit (0.3487 for stale startup).
This is not a torque-slew, contact/slip, disturbance or hardware qualification.

The nominal qpos/qvel/time arrays match the prior walking trace bit-for-bit.
Its known tracking failure therefore remains unchanged: legs RMSE 0.207371 rad;
pelvis-relative foot p95 0.266102 / 0.208334 m. Upright playback is not fidelity.
Only the EOF case executes the full walking input followed by a five-second
balance tail. Final joint-speed maximum is 0.006822 rad/s. The mid-walk fault
cases stop SONIC at the fault and ignore later source packets; they do **not**
complete the remaining source motion or demonstrate SONIC reacquisition.

## Implementation boundary

`gear_sonic/teleop/clocked_sim_session.py` requires one consecutive virtual
deadline per 20 ms. Missing, stale, malformed or noncontiguous input latches the
existing zero-velocity balance policy on that tick. Input returning cannot clear
the latch, and physics failure cannot silently restart/reset the simulation.
The forwarding observer preserves controller behavior while recording every
substep's commanded torque, measured joint position/velocity and simulation time.

The older live consumer blocks on packet receipt and excludes its subsequent
transport-hold interval from its optional trace. It was not edited or promoted.
This new driver does **not** open ZMQ, implement a real-time thread, reproduce a
live receiver's jitter or prove wall-clock schedulability. Its missing-packet
response is intentionally a new first-missed-deadline behavior, not the old
500-ms receive timeout rebranded as the same test.

All trials initialize at a source pose. Ordinary-standing acquisition, explicit
safe SONIC re-entry, broader motion tracking, real PICO capture and physical
Unitree ownership handback remain unfinished. The balance actor itself retains
29-to-23 compatibility mapping; these results do not prove firmware normal mode.

## Reproduce a new saved-input case

Run in the existing WSL environment. Choose a fresh output directory; overwrite
is refused. Other scenarios are `nominal`, `pause`, `gap`, `payload`, `stale-start`.

```bash
cd /mnt/z/codex/GR00T-WholeBodyControl-sonic-transfer-23dof
env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  PYTHONPATH=/mnt/z/codex/GR00T-WholeBodyControl-sonic-transfer-23dof:/mnt/z/codex/GR00T-WholeBodyControl/external_dependencies/unitree_rl_mjlab \
  /root/.venvs/g1_true23_mjlab/bin/python -m gear_sonic.scripts.record_g1_true23_clocked_input_sim \
  --repository-root /mnt/z/codex/GR00T-WholeBodyControl \
  --encoder-report artifacts/g1_true23_frozen_lora/paired_encoder_20260905_v2/original_breadth25/model_25.diagnostic.encoder.json \
  --decoder-report artifacts/g1_true23_frozen_lora/paired_encoder_20260905_v2/original_breadth25/model_25.diagnostic.decoder.json \
  --packets /mnt/c/Users/camer/sonic23_sim_artifacts/internet_pico_20260909_v1/walk002/causal_packets.json \
  --scenario end-of-stream \
  --output-directory artifacts/g1_true23_vr_clocked_manual_run_01
```

The command exits zero for its explicit scenario/physical screen only; every
report still has `tracking_qualified: false` and `deployment_ready: false`.

Focused regression: 27 tests passed in 12.41 s across clocked-session, existing
paired-live receiver and existing transport contracts; scoped Ruff E/F passed.

## Measured video

`end-of-stream/walk_then_balance.measured.mp4`: all 907 saved states, 50 Hz,
18.14 s including the initial frame, H264 960x720. No pose animation substituted
for dynamics and no extra physics performed by rendering. Camera follows root;
this video is not a world-position tracking comparison. Frames at 14.00 s and
18.12 s were visually inspected. Video SHA256:
`a633029929f5f04dfe6b806f48d25ae3f3eb384884f9f4d23d08d36615941f31`.

Next integration work must retain fixed-rate physics and complete traces while
adding explicit reacquisition and real receiver timing. Source tracking still
requires a separate controller/reference solution; do not relabel this fault
path PASS as full-body teleop or deployment completion.
