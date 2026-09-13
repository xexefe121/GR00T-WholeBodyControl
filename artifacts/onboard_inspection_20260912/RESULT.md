# Onboard access and controller inspection — September 12, 2026

Implementation update, September 13: Orin SSH worked earlier; the latest check
still reports both physical Windows Ethernet adapters Disconnected and WSL
exposes only Wi-Fi. See the timestamp in `latest_connection.json`.
No robot services or modes changed. Exact native23 factory dance/standing remains preserved. Newly
recovered native-leg locomotion physically completes received Pico and walk002,
but misses original full-body tracking/standing criteria. Full walk002 video is
complete. Completed dynamics pilots added full-range leg pose corrections while
upper-body targets remain directly tied to received poses. The previous learned
wrist-head recipe finished and is stopped. See [current implementation status](../teleop_resume_20260911/CURRENT.md).
Both bounded lifecycle pilots have finished
and stopped. GPU cumulative100/400 fail all four full motions. Exact native
MuJoCo/mjbatch checkpoints100/200 complete Pico plus the entire30s quiet hold,
but miss tracking and fail the three walking clips. Use `RUN_RECEIVED_NATIVE23_SIM.ps1` for the
experimental received-motion simulation. No recurring automation is active.

The pinned experimental Pico demo now completes160.6s including the entire30s
quiet hold under independent500Hz physics/50Hz control with zero deadline
misses. Run `RUN_PICO_NATIVE23_DEMO.ps1`; add `-Render` for its complete video.
Full-body tracking still fails, particularly foot placement. This is a concrete
fast full-duration simulation milestone, not a claim of general teleop readiness.
Both initial-velocity signs also complete the Pico lifecycle/quiet hold with
zero deadlines missed. The input-loss capture fix achieves quiet standing and
explicit rearm; its zero-spin run has one late physics tick. The100us spin trial
passes the input-loss scenario but worsens nominal timing. Zero spin is retained.
See [complete demo result](PICO_DEMO_RESULT.md) for the frozen controller and
all nominal, disturbance, input-loss, and remaining walking results.

Connected successfully to `unitree@192.168.123.164` with the user-authorized
manufacturer default login. This laptop uses temporary WSL Ethernet address
`192.168.123.223/24`; initial ARP probes found no owner. The other laptop, router,
robot network settings, and robot services were not modified. No motion or mode
commands were sent. Windows initially reported its existing `.200` as Duplicate;
the Windows administrator prompts were cancelled. Mirrored WSL can discard the
temporary address, so the local launcher checks and restores it after startup,
probing for conflicts before adding it.

## Concrete findings

- Accessible computer: NVIDIA Orin NX, Ubuntu 20.04.6, L4T R35.3.1. Installed
  Unitree package manifests identify their modules with the `pc4` suffix.
- The factory locomotion service responds through the installed SDK:
  `GetFsmId` returned success (`0`) and data `801`. This was a read-only query;
  it does not establish any tracking, balance, or teleop performance.
- The installed G1 client exposes velocity, standing-height, balance, and preset
  task commands. The inspected client does not expose arbitrary full-body pose
  tracking. Its two source files were copied under `extracted/`.
- No factory locomotion or full-body policy weights were found in the bounded
  inspected directories. Those include `/unitree`, `/home/robot_emb`,
  `/home/unitree/unitree`, `/opt/ota_package`, the PC4 installer, and the existing
  `g1_true23_onboard` deployment directory. This is not proof that no such files
  exist elsewhere or inside binaries.
- Model files in `/home/robot_emb` are named YOLO vision models. Its
  `robot-ai.service` launches a Uvicorn web application. The service name is not
  evidence that it contains Unitree's factory AI locomotion policy.
- Existing SONIC `policy/release` contains three observation YAML files and no
  policy weights. The stored deployment code is not a working factory-policy
  export.
- TCP port 22 on the expected control-computer address `192.168.123.161` refused
  connections, including a check from the robot's Orin. No alternate passwords,
  exploits, mode changes, or SSH-enabling changes were attempted.

## PC1 and the factory-policy breakthrough

PC1 at `192.168.123.161` responds to factory SDK queries. Its SSH port remains
closed. Port 9991 is open; no WebRTC control session was established. Read-only
SDK calls returned active mode `ai`, FSM 801, 30 service entries, and an arm-action
catalog containing preset gestures and named custom arm sequences. That catalog
alone is not evidence of accessible full-body dance weights. PC1 filesystem and
its installed ai_sport version remain unverified.

Instead of altering PC1, downloaded Unitree's public G1 Edu+ 1.4.5 firmware:
[official firmware](https://unitree-firmware.oss-cn-hangzhou.aliyuncs.com/firmware/release/package_1.4.5.0_G1_Edu%2B_1759976671033.upk).
Extracted ai_sport 8.4.2.222 entirely locally. It contains 107 MNN network files,
14 ONNX network files, 82 CSVs and 63 encoded YAMLs. All MNN graphs loaded on this
Windows laptop. All 63 YAMLs were recovered and their stored checksums verified.
Network file count includes alternative versions and paired encoders; it is not
a count of independent skills.

The native23 mimic config explicitly names the correct joint layout: 12 leg
joints, waist yaw and five joints per arm, with SDK indices
`0..12,15..19,22..26`. Its deployed 9.35-second dance model now runs in the native
MuJoCo 3.2.3 referee, with 500 Hz physics and 50 Hz policy evaluation.

### Reproduced result

`run_factory_mimic_sim.py` implements the recovered observation/action contract.
The first reproduction completed the dance and 30-second standing hold but
exceeded right ankle roll by 0.0466 rad. A uniform 0.06-rad inward target margin
plus a generic restoring/braking torque inside the joint-limit band eliminated
that overshoot. These modifications are recorded; this is not an unchanged
vendor-binary claim. Joint ranges, speed limits, effort limits and physical
states were not relaxed or projected.

The same modified controller completed all three 42.36-second runs (3 seconds
initial stand, 9.35-second dance, at least 30 seconds standing): nominal and
initial root x velocities +0.03 and -0.03 m/s. All three had zero joint-bound
excess, stayed below native speed/effort limits and had no falls or engine
warnings. Nominal median inference was 0.105 ms, maximum 0.448 ms. Last-three-
second joint speed maximum was about 0.00037 rad/s. No training was required.

This is a fast full-motion-and-standing factory baseline, **not simulation-ready
live teleop**. The policy accepts motion phase, not arbitrary received body poses.
The four teleop recordings, tracking thresholds, movement input-loss/rearm, real
Pico input, and independent wall-clock scheduling have not been passed with this
factory controller. Its native dynamics test was unpaced; inference timing alone
does not certify deadlines.

Run `RUN_FACTORY_NATIVE23_SIM.ps1`; add `-Render` for the full-duration video.
This launcher has no robot networking or DDS commands.

### Concrete next controller route

Keep this factory23 controller as a proven balance/standing baseline and source
of physical expert trajectories. Reconstruct the separate `cpy_dance`/`cpy_fight`
pose-conditioned interface next: those graphs take memory [465], proprioception
[93], current quaternion [4], and target state [71], returning 29 actions. The
paired factory trajectories and configuration are available. Their 29-joint
output must be tested under the six missing joints in native23; dropping six
outputs is not evidence of a successful transfer. First decisive evaluation is
received-pose movement followed by standing on native23, before more training.

Local firmware root:
`E:\codex-artifacts\sonic23_teleop_resume_20260911\onboard_factory_firmware_v1`.
Nominal report/video: `native23_mimic_brake_v1/`.
Perturbation reports: `native23_mimic_brake_push_plus/` and
`native23_mimic_brake_push_minus/`.
Graph interfaces: `mnn_graph_catalog.json`, `onnx_graph_catalog.json`.
Recovered config/checksums: `decoded_configs/decode_manifest.json`.
The source is a public firmware candidate; it has not been matched to this
robot's installed PC1 firmware.

## Reconnect without changing the other laptop

Run `RUN_ONBOARD_INSPECTION.ps1`; it performs only software inventory reads over
SSH. Optional `-Stage controller` or `-Stage factory` repeats the bounded file
metadata inspection. Uses the existing WSL SSH host-key record with strict
verification. Requires the already-present WSL `sshpass` and root local networking
capability. No Windows administrator prompt is required.

Evidence: `pc2_inventory.json`, `pc2_controller_details.json`,
`pc2_factory_details.json`, `pc2_reuse_files.json`, and
`factory_interface_result.json`, `pc1_network_access.json`,
`pc1_service_catalog.json`. No training or scheduled task was started.
