# Selected true23 frozen-LoRA live PICO test

This runbook starts the selected 23-DoF policy in **CPU MuJoCo only**. The
consumer is restricted to localhost, opens no DDS or Unitree channel, and
publishes no robot commands. A timeout, stale packet, 50 Hz gap, malformed
payload, physical velocity jump, or excessive tilt latches the reviewed
zero-velocity balance fallback.

The selected decoder is fixed at SHA-256
`44d1fb2701f1e65460f1c2c23f676bce4f1d4a44b3b112798dc5034af37946b8`.
The runtime also pins its decoder report and parity summary. Substituting any
of those three files fails before the ZMQ subscriber opens.

## Qualification already completed

The authentic saved PICO `walk001` clip was published through the exact live
localhost ABI at 50 Hz. The selected candidate completed all 684 transitions,
frames 10 through 693, with 30 ms maximum reference age, 0.7126 m minimum base
height, 0.3399 rad maximum tilt, and no fallback. The original true23 live run
also completed 684/684, with 24.77 ms maximum age, 0.6706 m minimum height,
0.3648 rad maximum tilt, and no fallback.

Timeout, missing-frame, and stale-frame injections were each applied after
120 live transitions. Every case latched the expected transport trigger and
held the balance fallback for 100 transitions. The fallback stayed above
0.748 m and below 0.160 rad tilt. The machine-readable audit reports
`software_live_teleop_ready: true` and original transport parity true.

## Before putting on the headset

Start XRoboToolKit on PICO, pair and calibrate the body trackers, and use a
clear area. Then run this read-only WSL health probe from the isolated
worktree:

```bash
cd /mnt/z/codex/GR00T-WholeBodyControl-sonic-transfer-23dof
PYTHONPATH=.:/mnt/z/codex/GR00T-WholeBodyControl/external_dependencies/XRoboToolkit-PC-Service-Pybind_X86_and_ARM64 \
  /usr/bin/python3 -m gear_sonic.scripts.probe_g1_true23_pico_tracking_health \
  --duration-seconds 5 \
  --output artifacts/g1_true23_frozen_lora/live_teleop_v1/pico_health_live.json
```

Do not continue unless the report says `passed: true`. The latest probe on
this workstation verified the service and binding hashes but saw no connected
headset or trackers, so that external gate is currently closed.

## Live PICO to MuJoCo

Use two terminals. Start the consumer first.

**Terminal 1 — Windows, selected policy consumer:**

```powershell
cd Z:\codex\GR00T-WholeBodyControl-sonic-transfer-23dof
python -m gear_sonic.scripts.run_g1_true23_frozen_lora_live_teleop `
  --repository-root Z:\codex\GR00T-WholeBodyControl `
  --decoder-report artifacts\g1_true23_frozen_lora\original_sonic_happy_residual_v1\candidate.plus_0p002.decoder.json `
  --candidate-summary artifacts\g1_true23_frozen_lora\original_sonic_happy_residual_v1\candidate.plus_0p002.summary.json `
  --endpoint tcp://127.0.0.1:5557 `
  --steps 1500 `
  --startup-timeout-ms 120000 `
  --receive-timeout-ms 500 `
  --maximum-age-ms 100 `
  --fallback-hold-steps 100 `
  --viewer `
  --output artifacts\g1_true23_frozen_lora\live_teleop_v1\pico_live_30s.json
```

**Terminal 2 — WSL, real PICO producer:**

```bash
cd /mnt/z/codex/GR00T-WholeBodyControl-sonic-transfer-23dof
PYTHONPATH=. /usr/bin/python3 -m gear_sonic.scripts.stream_g1_23dof_pico_causal_zmq \
  --workspace /mnt/z/codex/GR00T-WholeBodyControl-sonic-transfer-23dof \
  --xrt-module-dir /mnt/z/codex/GR00T-WholeBodyControl/external_dependencies/XRoboToolkit-PC-Service-Pybind_X86_and_ARM64 \
  --soma-source-root /root/.cache/g1_true23_soma/source \
  --bind tcp://127.0.0.1:5557 \
  --packets 1500 \
  --timeout-seconds 120 \
  --subscriber-warmup-s 2 \
  --frame-timeout-s 2 \
  --pico-client-apk-sha256 e4ac54e057eb5984eb623e43e659678892fb687b3088b29f2e26116eda94b7c9 \
  --evidence artifacts/g1_true23_frozen_lora/live_teleop_v1/pico_live_30s.producer.jsonl
```

The consumer succeeds only after all 1500 fresh, contiguous transitions. If
the producer stops or tracking stalls, MuJoCo immediately leaves SONIC motion
tracking and holds the fallback policy.

## Physical G1 boundary

The simulator consumer above remains read-only. Do not connect it to a robot
publisher. Physical control uses the separate promoted C++ controller plus a
fresh hardware shadow and gantry-only active sidecar.

Two managed Windows launchers now own the complete producer/controller
lifecycle:

- `run_g1_true23_frozen_lora_dance_gantry` replays the hash-bound original
  SONIC dance clip. It repeats source values with new contiguous source indices
  and exact 20 ms timestamps, so the READY window does not expire.
- `run_g1_true23_frozen_lora_live_gantry` starts the real PICO/SOMA causal
  producer first. It does not start the robot controller until publisher
  evidence contains a fresh reference packet. Publisher loss sends SIGINT to
  the controller and ends in fail-safe damping.

Both launchers keep the controller-attached console visible. They do not
auto-arm. After `[READY]`, hold L2 and press A once. L2 release, B/R2, app or
physical e-stop, stale policy, state loss, source loss, a joint limit, or a
command-write failure stops policy motion and writes the full reviewed damping
tail. `[REMOTE]` lines show every decoded L2, A, and STOP transition, so a
missing wireless edge is visible before claiming that the robot armed.

Example saved-dance launch after creating a fresh sidecar:

```powershell
python -m gear_sonic.scripts.run_g1_true23_frozen_lora_dance_gantry `
  --repository-root Z:\codex\GR00T-WholeBodyControl-sonic-transfer-23dof `
  --encoder Z:\codex\GR00T-WholeBodyControl\artifacts\g1_true23\causal_model_250_20260803\causal_model_250.encoder.onnx `
  --decoder-report artifacts\g1_true23_frozen_lora\original_sonic_happy_residual_v1\candidate.plus_0p002.decoder.json `
  --promotion artifacts\g1_true23_frozen_lora\physical_dance_v1\candidate.plus_0p002.dance_shadow_promotion.v2.json `
  --active-promotion <fresh-active-sidecar.json> `
  --live-shadow-evidence <fresh-shadow.jsonl> `
  --packet-bundle artifacts\g1_true23_frozen_lora\physical_dance_v1\original_sonic_happy.true23.causal_packets.json `
  --authorization-id <matching-id> `
  --evidence <new-controller-evidence.jsonl> `
  --publisher-evidence <new-publisher-evidence.json> `
  --duration-seconds 5 `
  --repeat-count 100 `
  --gantry-authorize I_CONFIRM_G1_TRUE23_STAGE1_GANTRY
```

For live PICO, use `run_g1_true23_frozen_lora_live_gantry` with the same
artifact arguments plus `--xrt-module-dir` and the installed hardened PICO APK
SHA-256. Its pinned publisher Python defaults to the SOMA venv containing
`pyzmq`; its raw capture worker remains `/usr/bin/python3` for the XRT binding
ABI. A live-health probe on 2026-09-01 still reported no connected headset,
trackers, calibration, or body stream. Saved-clip physical shadow passed, but
live-headset physical teleoperation and an armed dance remain unproven until
those external inputs are present and terminal execution evidence passes.
