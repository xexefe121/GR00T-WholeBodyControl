# True23 incident capture — passive transport, offline interpretation

Status, 2026-09-07: **diagnostic tooling, not deployment qualification**.
No physical robot, DDS participant, SSH connection, motor command or mode
request was used to implement or test this tooling. The local robot Ethernet
adapter remained disconnected. Existing hardware control edits are unchanged.

The reproduced restore-RPC defect and the separate unresolved physical
motors-off incident are documented in `G1_TRUE23_HANDOFF_FINDINGS.md`.
This recorder addresses the missing fault-time evidence. It does not repair
ownership transfer or make the native23 dance policy complete its motion.

## What gets preserved

`gear_sonic/scripts/record_g1_true23_incident_readonly.py` subscribes to:

- `rt/lowstate`: HG state, including all 35 packet slots.
- `rt/lowcmd` and `rt/user_lowcmd`: commands published by other processes.
- `rt/api/sport/request` and `rt/api/sport/response`.
- `rt/api/motion_switcher/request` and `rt/api/motion_switcher/response`.

There is no command publisher, RPC client or automatic motion launch. DDS
discovery/control traffic still exists; “read-only” means no application motor
or mode commands. Unpublished or inaccessible topics cannot be reconstructed.
The tool never claims that observing a command proves the robot applied it.

Each accepted callback snapshots the SDK message into reserialized CDR bytes,
then sends that immutable snapshot to a bounded writer queue. Records include
a receipt index, local monotonic timestamp, topic, raw CDR base64 and SHA256.
Timestamp/index assignment occurs under the callback lock; this is neither a
source clock nor wire-arrival timing. Host, boot ID, UTC start, interface,
source/IDL/native-CRC hashes and before/after source checks accompany capture.

The default queue holds at most 2,048 records, each with at most 65,536 payload
bytes. Oversize/serialization failures and queue drops are counted explicitly.
The SDK's additional user queue is disabled. DDS/network losses before our
callback remain unknown; zero collector drops does not exclude upstream loss.
Existing output directories are rejected rather than overwritten.

Full decoding is **not performed during live capture**. The initial design
decoded/CRC-checked/expanded JSON concurrently and dropped 4,365 of 10,000
synthetic packets in its paced test. Preserved failure evidence is under
`artifacts/g1_true23_frozen_lora/incident_capture_20260907_v1/benchmark10/`.
Increasing the queue alone would only defer that sustained throughput failure.

After recording, `gear_sonic/scripts/decode_g1_true23_incident_no_robot.py`
verifies the saved container and each payload hash, checks receipt ordering
and counts, then creates a separate `decoded.jsonl.gz` and decode summary.
The original recording is not modified. It decodes real SDK IDL and calculates
native CRC, keeping raw status/mode bits, joint position/velocity/torque,
voltage/temperature, IMU, remote bytes and RPC identities/status/payloads.
Unexpanded fields remain present in CDR. Nonfinite floats are represented
explicitly in valid JSON instead of causing a frame to disappear.

Observational edges use only the 23 controlled motor slots. Absent body axes
13, 14, 20, 21, 27 and 28 and reserved slots 29–34 remain in the raw recording
but are not treated as controlled motors. Bad CRC, malformed state, repeated
or reset tick, and reversed local time break the edge baseline; uint32 tick
wrap is supported. Known collector gaps are attached to observed changes.
A raw bit-30 change is reported without assigning its physical cause.
`mode_machine` is a machine-configuration field, **not the locomotion FSM**.
Raw `version` fields do not establish firmware/API compatibility.

## Use boundary

Run observation on the external control/observer computer, not as an added
load on the robot's real-time control computer. First verify the observer's
actual robot-facing interface and routing. The Ethernet-disconnected state
observed during this work is not a working live capture setup.

On this workstation, the tested codec environment is WSL Ubuntu-22.04,
`/usr/bin/python3` (3.10), with the existing SDK at:

`/mnt/z/codex/GR00T-WholeBodyControl/external_dependencies/unitree_sdk2_python`

The training venv does not have Cyclone DDS installed. Do not confuse it with
the recorder environment or install new robot-side software for this check.
Use the separate repository root:

`/mnt/z/codex/GR00T-WholeBodyControl-sonic-transfer-23dof`

From that root, after supplying the verified interface, the passive command is:

```bash
env PYTHONPATH=/mnt/z/codex/GR00T-WholeBodyControl-sonic-transfer-23dof:/mnt/z/codex/GR00T-WholeBodyControl/external_dependencies/unitree_sdk2_python \
  python3 -m gear_sonic.scripts.record_g1_true23_incident_readonly \
  --interface REPLACE_WITH_VERIFIED_INTERFACE \
  --duration-seconds 30 \
  --output-directory artifacts/g1_true23_frozen_lora/incident_NEW_CAPTURE
```

`--multicast-local-ip` is optional and requires a verified IPv4 address on that
same observer interface; it only adds multicast membership. Do not guess the
address. The tool neither connects via SSH nor queries robot firmware.

After capture ends, offline interpretation is:

```bash
env PYTHONPATH=/mnt/z/codex/GR00T-WholeBodyControl-sonic-transfer-23dof:/mnt/z/codex/GR00T-WholeBodyControl/external_dependencies/unitree_sdk2_python \
  python3 -m gear_sonic.scripts.decode_g1_true23_incident_no_robot \
  --capture-directory artifacts/g1_true23_frozen_lora/incident_NEW_CAPTURE \
  --output-directory artifacts/g1_true23_frozen_lora/incident_NEW_CAPTURE_decoded
```

Capture exit 0 means recording finished, accepted callbacks were preserved
without collector errors/drops, and at least one LowState packet was saved.
It does not even imply that that state's CRC is valid. Interpret the data
afterward. Decoder exit 0 means source capture was complete and decoding had
no processing errors; CRC failures remain explicit observations. Neither exit
code proves healthy motors, normal standing, firmware support, transport
ownership or safety. Every report keeps `deployment_ready=false` and
`hardware_authorized=false`. Interrupting recording only stops observation;
it cannot stop an independently running robot controller.

## Verification and remaining blockers

The focused suite passes **73 tests, zero skips** using the real SDK codec
dependency plus fake subscriptions with DDS/network constructors forbidden.
It covers raw-copy ownership, both command topics, real RPC IDL, native CRC,
nonfinite values, bounded overflow, transient raw status edges, absent slots,
tick wrap/reset, source/payload tampering, cleanup failures and output
exclusivity. The vendored SDK's broken default RequestLease/Response factories
were exposed by initial test attempts; tests now use its actual IDL constructors,
as its RPC implementation does. No SDK code was patched.

The raw-first 10-second test preserved all **10,000 callbacks** at a requested
500 state + 500 command packets/s and recovered both edges of a one-sample
synthetic bit-30 pulse. Queue peak was 318/2,048; callback p99 was 0.717 ms,
but the maximum was 116.677 ms. That scheduling pause prevents treating this
as a real-time or DDS reliability qualification. Offline interpretation took
9.367 seconds after recording closed. The benchmark creates IDL objects only;
it does not publish synthetic packets onto DDS.

Evidence root: `artifacts/g1_true23_frozen_lora/incident_capture_20260907_v1/`.
`raw_first_tests_v1.xml` is the 73-test JUnit record; `raw_first10/benchmark.json`
contains the initial raw-first benchmark and exact code hashes. The final
`raw_first30/benchmark.json` preserves all **30,000 callbacks** over 30 seconds,
with queue peak 335, callback p99 0.716 ms, maximum 2.082 ms and 27.066 seconds
of offline decoding. Both synthetic status edges are recovered. This better
scheduling result does not erase the earlier host pause or establish a bound.
The final manifest is `verification.json`, SHA256
`d56c00dd9f76cfb9f4f8bab263621105b415f311d2c1e7be662fa1c03c7f0d18`.
Do not treat older source hashes as a qualification of subsequently changed code.

Before any physical motion, current firmware support and a supported normal
standing ownership handoff still need to be established. The candidate
retained-service `rt/user_lowcmd` protocol is not yet qualified for this robot.
Policy qualification is separately incomplete: historical-start dance reaches
2.06 of 10.7 seconds and has no successful full standing return. This tool does
not change those results, authorize a dance, or provide 29-axis physical parity.
