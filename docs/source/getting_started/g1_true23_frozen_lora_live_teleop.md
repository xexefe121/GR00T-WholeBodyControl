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

**Current physical status: not ready for live teleop.** A dance session that
returns the G1 in damped mode fails the same ownership/shutdown path used by
teleop. No further physical dance or teleop run is a valid readiness test until
the robot-free lifecycle gate below passes and a later bounded hardware handoff
test proves the robot remains in the captured standing mode.

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
  evidence contains a fresh reference packet. Publisher loss starts a
  positive-gain posture return and exact Unitree mode handoff.

Both launchers keep the controller-attached console visible. Saved-clip dance
uses a separately bound exact `DANCE` command and starts automatically only
after `[READY]`; it does not depend on L2/A state. Direct mode is restricted to
the hash-bound frozen-LoRA dance, gantry-only, and at most five seconds.
Reviewed duration, wireless B/R2 or L2 release, and app/process cancellation
stop policy motion without dumping the robot: the controller snapshots the
current 23-joint pose, writes 250 positive-gain zero-feedforward hold packets,
restores the exact Unitree motion mode captured before release, verifies that
mode is active, and only then closes LowCmd. Stale policy/state/source and
inference faults use the same return. If the command writer itself fails, the
main thread immediately re-selects and verifies the captured Unitree mode.
Final-boundary validation rejects every post-release command whose controlled
joints lack positive `kp`/`kd`; the controller never publishes a synthesized
`kp=0` damping tail. Physical e-stop remains the independent hard-stop path.

Real PICO live teleop retains the wireless deadman contract. After `[READY]`,
hold L2 and press A once. `[REMOTE]` lines show every decoded L2, A, and STOP
transition for that live path.

Controller startup is hold-first. ONNX and the ten-frame real-proprio window
warm while Unitree motion mode still owns posture, with zero LowCmd writes. The
LowCmd publisher is initialized without writing, the current 23-joint posture
is sampled, and only then is motion mode released. The first post-release
packet has positive position gains and zero feedforward torque. At least 25
successful posture-hold packets must be written before direct or wireless
arming becomes possible. `kp=0` damping is fault-only; successful dance or
teleop completion must prove positive-gain return hold and motion-mode restore.
Execution evidence rejects any pre-release write, any startup damping packet,
or a first hold packet delayed by more than 20 ms after release.

The 2026-09-01 physical records
`live_dance.direct.execution.b0180a162573.jsonl` and
`live_dance.direct.verify.execution.b0180a162573.jsonl` are invalidated for
readiness. Their controller released motion mode and emitted `kp=0` damping for
about 22.75 seconds while policy history warmed, matching the observed dumped
mode. Their later policy-packet counts do not prove a safe startup. Do not use
either record as physical qualification.

Later physical dance attempts that ended with the robot damped are also failed
evidence. They exposed two remaining lifecycle bugs: a simultaneous
watchdog/end-of-clip transition was misread as failed return and latched
`OperatorStop`, and the writer exception handler intentionally emitted 250
damping packets. Both paths are now removed, but this code change alone does
not make live teleop physically ready.

The bounded one-second `lifecyclefix_1s_20260902_030328` run used the rebuilt
no-damping binary. It wrote 467 policy packets, then 305 positive-gain return
packets, with zero damping packets and zero rejected non-positive-gain packets.
It still failed qualification because `SelectMode("ai")` ran while the LowCmd
writer remained active, and Unitree rejected the ownership-transfer RPC. The
follow-up interlock now requests handoff, waits for the writer to quiesce,
closes the LowCmd publisher, retries `SelectMode`, and requires exact
`ai / 801 / 0` before success. That change has robot-free coverage only; no
later physical run has validated it.

Run the no-robot lifecycle qualification from WSL after rebuilding
`true23_active_gantry_core_harness` and `g1_true23_active_gantry`:

```bash
cd /mnt/z/codex/GR00T-WholeBodyControl-sonic-transfer-23dof
python3 -m gear_sonic.scripts.qualify_g1_true23_active_lifecycle_no_robot \
  --repository-root . \
  --output /tmp/g1_true23_active_lifecycle_no_robot.json
```

This command launches neither controller nor publisher. It opens no DDS or
LowCmd channel. Pass requires nine lifecycle scenarios, 4,000 positive-gain
recovery frames, injected 101/290/500 ms stalls, zero published damping frames,
exact mode/FSM matching, and a compiled/source surface audit that forbids the
old damping-tail code.

Future physical readiness order is strict:

1. Robot-free lifecycle report passes.
2. Read-only PICO health and hardware shadow pass; no LowCmd.
3. Separate bounded ownership-handoff smoke proves captured standing mode
   returns after positive-gain hold.
4. One-second then five-second gantry dance pass with zero damping packets and
   exact mode/FSM restore in execution evidence.
5. Only then run one-second live PICO teleop with deadman, followed by longer
   sessions.

The original `GR00T-WholeBodyControl` true23 implementation has the same
startup defect: it releases motion mode and starts the writer before policy
readiness, and its unarmed `BuildCommand` path returns `BuildDampingCommand`.
This separate sonic-transfer repository intentionally diverges there. It adds
policy-before-release prewarm, positive-gain sampled hold, delayed operator
arming, hold-first evidence gates, and distinct direct-versus-wireless frozen
SONIC active sidecars. Policy/action mapping remains the same true 23-DoF
mapping and safe-target transform.

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
  --gantry-authorize I_CONFIRM_G1_TRUE23_STAGE1_GANTRY `
  --direct-dance-command DANCE
```

For live PICO, use `run_g1_true23_frozen_lora_live_gantry` with the same
artifact arguments, but create the active sidecar with
`authorize_g1_true23_frozen_lora_live_gantry`. The live sidecar is distinct from
the direct-dance sidecar: it binds wireless L2/A and B/R2, a 1--10 second
reviewed window, and no direct `DANCE` command. Add `--xrt-module-dir` and the
installed hardened PICO APK SHA-256. Its pinned publisher Python defaults to the SOMA venv containing
`pyzmq`; its raw capture worker remains `/usr/bin/python3` for the XRT binding
ABI. A live-health probe on 2026-09-01 still reported no connected headset,
trackers, calibration, or body stream. Saved-clip physical shadow passed, but a
hold-first armed dance and live-headset physical teleoperation remain unproven
until new terminal execution evidence passes.
