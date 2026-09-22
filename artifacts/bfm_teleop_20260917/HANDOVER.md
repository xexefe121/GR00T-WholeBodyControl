# Handover: BFM-Zero whole-body teleop on the Unitree G1

Written for the next engineer or agent picking this up cold, on 2026-09-23.
Read this before `PROGRESS.md`. `PROGRESS.md` is the full dated lab notebook,
including conclusions that were later corrected; this file is the current
state.

## What exists

A 23-DoF whole-body teleop stack for a Unitree G1 rev-1.0 (`mode_machine == 4`),
built around the BFM-Zero policy. A motion source produces reference packets, a
50 Hz policy process on the Windows workstation turns them into joint targets,
and a native 500 Hz loop on the robot's Jetson Orin owns the LowCmd stream, the
bring-up ladder and every safety abort.

    motion source ──► BFM policy (Windows, 50 Hz) ──► native loop (Orin, 500 Hz) ──► rt/lowcmd
      recorded clip                                    ladder + aborts + estimator
      or live PICO

Safety decisions live on the robot, deliberately: the Ethernet link between
workstation and robot has a long tail (target age reached 62.9 and 74.9 ms,
round trip 132 and 156 ms, in otherwise clean runs), so the robot must be able
to bring itself down without hearing from the PC. Losing the operator heartbeat
for one second aborts locally.

## Status

**Proven on the physical robot.**

| | |
|---|---|
| Full walk002 clip, robot standing on its own feet | 1,417 of 1,417 targets, 0 deadline misses, run ended only because the clip ran out |
| Target age during that run | 1.07 / 1.36 / 6.93 ms against a 100 ms bound |
| Bring-up ladder | observe → zero torque → damping → position hold → default pose → policy, each on an explicit operator command |
| Standing up | knee 0.93 → 0.39 rad under its own torque; tilt improved 9.6° → 2.5° |
| Real torque under BFM | knee −23.4 Nm, hip +10.0 Nm, ankle +7.9 Nm |
| Native loop timing under real robot load | 0 misses, LowCmd gap p50 2.0 ms, max 2.6 ms |
| Aborts seen firing correctly on hardware | operator liveness, position error, tilt, joint velocity, stale target |

**Proven against recordings or in simulation only.**

- The live PICO → BFM bridge: 6,541 of 6,541 packets admitted by the unchanged
  gate, zero validation failures, capture-to-publication p95 15 ms, minimum
  foot clearance corrected from −58.2 mm to +1.6 mm with nothing published
  below the floor.
- The estimator's agreement with its Python reference (1.1e-14) and the
  emitted-target equivalence (exact).

**Not proven.**

- The live headset path has never run with a person wearing the PICO. Live
  capture timing, tracking dropouts, real end-to-end latency and live packet
  admission are all unmeasured.
- The full 6,541-frame PICO-derived motion fails in simulation at control
  5,130 with a 0.0124 rad range excess. Short segments complete. Do not assume
  a long live session will hold.
- How long a clip session sustains is unknown: the only long run ended because
  the clip finished, and an earlier one because the operator pressed the stop.

## Running a clip session

Everything below assumes the robot is in its gantry with the harness able to
catch it, and a person standing at the physical stop.

1. **After any reboot, power cut or source change**, restore the robot side:

       powershell -File gear_sonic\scripts\g1_true23_robot_rebuild.ps1

   This copies the loop and tool sources, builds them on the robot, restores
   the binary's capabilities and the real-time grant, and verifies all of it.

2. **Put the robot into debug mode by hand.** Nothing actuates without this.
   Suspended or standing, first damping (the operator uses the E-stop; `L2+B`
   is the documented route), then **`L2+R2`**. `L2+A` poses a diagnostic
   position and is the visible confirmation it took.

3. **Run the session:**

       powershell -File gear_sonic\scripts\run_g1_true23_live_teleop.ps1 -Clip walk002 -PolicyBrakeStep 0.02

   It checks the deployment, restores the real-time grant, starts the operator
   client *before* arming, climbs the ladder, proves the robot is actuating
   before commanding any motion, reaches the default pose, starts BFM and the
   clip behind a readiness barrier, hands over, prints the stage every ten
   seconds, and stops everything cleanly on exit or Ctrl+C.

`-PolicyBrakeStep` is the per-control step for the policy stage only. 0.02
completed a full clip standing; 0.035 ran a suspended robot; the qualified cap
is 0.100. Lower it for gentler motion. **Do not raise the velocity limit
instead** — it has caught real events.

## Running the live PICO path

The PC side is installed and verified: XRoboToolkit's RoboticsService runs in
WSL, the XR receiver listens on 60061, WSL is in mirrored networking so the
headset reaches the workstation's own address, and the capture chain starts and
runs as far as waiting for tracking.

On the headset: strap a Motion Tracker to each ankle and power both on; wear
the headset with both controllers on; open the hardened XRoboToolkit client;
connect to the workstation's Wi-Fi address (**192.168.1.182** at time of
writing — check it, DHCP may move it); select **Full body / BodyTracking**, not
raw MotionTracking, which is a mutually exclusive mode; complete calibration;
leave the client in the foreground.

Then:

    powershell -File gear_sonic\scripts\run_g1_true23_live_teleop.ps1 -Source pico -PolicyBrakeStep 0.02

`-Source pico` starts the policy, then the bridge, then the headset publisher,
in that order — started the other way round the policy misses the first packets
and the gate records a sequence-gap fault. It fails fast with a clear message
if no targets arrive.

**Do this in simulation first.** The live path has never driven the robot. Run
the bridge into the simulation consumer, confirm the robot in MuJoCo follows
the wearer, and only then attach it to hardware.

## Where things live

| thing | path |
|---|---|
| Native 500 Hz loop, ladder, aborts, estimator | `gear_sonic_deploy/src/g1/g1_deploy_onnx_ref/src/g1_true23_bfm_lowcmd_loop.cpp` |
| Robot diagnostic tools and what each answers | `gear_sonic_deploy/src/g1/g1_deploy_onnx_ref/tools/` |
| 50 Hz BFM policy process | `gear_sonic/scripts/run_g1_true23_bfm_split_policy.py` |
| Operator client (advance, heartbeat, manual abort) | `gear_sonic/scripts/run_g1_true23_bringup_operator.py` |
| Ladder reference implementation (the specification) | `gear_sonic/utils/g1_true23_bringup.py` |
| Live PICO → BFM bridge | `gear_sonic/scripts/stream_g1_true23_pico_bfm_packets.py` |
| PICO capture and pinned solver (upstream) | `gear_sonic/scripts/stream_g1_23dof_pico_causal_zmq.py` |
| One-command session | `gear_sonic/scripts/run_g1_true23_live_teleop.ps1` |
| Robot rebuild | `gear_sonic/scripts/g1_true23_robot_rebuild.ps1` |
| Real-time grant only | `gear_sonic/scripts/orin_enable_realtime.sh` |
| Operator procedure | `artifacts/bfm_teleop_20260917/HARDWARE_BRINGUP.md` |
| Full history | `artifacts/bfm_teleop_20260917/PROGRESS.md` |

Robot deployment lives at `/home/unitree/bfm_teleop_fix5` (source, build, tools,
runs). `/home/unitree/g1_true23_onboard` is a protected checkout: read it for
the SDK, never write to it.

Ports: robot binds 5910 state, 5911 targets, 5912 operator control. On the
workstation the clip publisher uses 6070 and the PICO bridge 5591; the PICO
upstream binds 5557. The loop writes live status to
`/dev/shm/g1_bringup_status.json` twice a second — stage, abort reason, target
age, operator frames, misses. Read that during a run rather than waiting for
the end-of-run report, which a killed run never writes.

## Things that will waste your day

**The robot's E-stop cuts power.** It reboots the Orin, and an unclean shutdown
has deleted the freshly linked loop binary. After any stop: run the rebuild
script. Prefer the software abort; it ramps to damping then zero torque.

**Debug mode is the gate on all actuation.** Without it the robot accepts every
LowCmd and applies no torque, and looks identical to a working system until a
command asks a joint to move. Position hold cannot reveal it — it commands the
pose the robot is already in. Use `tools/twitch`; the session script does this
automatically before it commands motion.

**Every reboot clears the real-time grant** (`user.slice` `cpu.rt_runtime_us`,
kernel built with `CONFIG_RT_GROUP_SCHED`), and **every rebuild drops the
binary's file capabilities**. Without them the loop is starved by the robot's
own SLAM stack: 230 missed deadlines and 25 ms of starvation in one measured
run, versus zero with them.

**`/tmp` on the robot is wiped on reboot.** Keep tools in
`/home/unitree/bfm_teleop_fix5/tools`.

**Never pass `$` expressions inline through `wsl.exe` from Windows or Git
Bash** — the outer shell eats them and you get empty paths and confident wrong
conclusions. Send remote logic base64-encoded, as the scripts here do.

**`pgrep` without `-f` cannot match this binary's name** (too long); it will
report the loop as not running while it is.

**CPU affinity is counterproductive on the Orin.** Unpinned: zero misses.
`taskset -c 7`: 20 misses. `taskset -c 6,7`: 1,471 misses.

**195 tracked artifact files sit behind symlinks** to `N:\codex-archive`
(`artifacts/g1_true23_frozen_lora`, `artifacts/g1_true23_generalist`, moved when
`Z:` filled). They are marked `skip-worktree` in this clone so they stop
appearing as deleted. If you clone fresh, expect them to appear deleted again —
do not commit that, and never `git add -A` here.

**The operator client must be running before the loop arms**, or the
one-second deadman aborts within a second.

**Start the consumer before any publisher.** A publisher that streams while the
policy is still building its model fills the void: the gate never sees a packet
and the policy correctly emits nothing.

## Open work, in the order I would do it

1. **Drive the live PICO path in simulation with a person wearing the headset.**
   Everything else is ready; this is the only way to learn what live capture
   actually does.
2. **End a clip cleanly.** When the source is exhausted the run currently ends
   on a stale-target abort, which is correct but reads as a failure. The packet
   envelope already carries `final`; use it to stop and hold.
3. **Investigate the full-motion simulation failure** at control 5,130
   (0.0124 rad range excess) before any long live session.
4. **Look at arm tracking through the bridge.** The same walk002 clip gives arm
   RMSE 0.043 direct and 0.125 through the bridge, while leg and root improve.
   Three times worse on identical input suggests something in the body mapping.
5. **Find out how long a session sustains.** No run has ended on its own yet.

## Rules worth keeping

Do not relax a safety bound to make a run succeed. Every limit here has caught
something real: the position-error abort caught a 155° twisted torso, the tilt
abort caught a suspended robot swinging, the velocity abort caught a foot
landing. Where a limit fired on a normal event, the fix was to stop the command
outrunning the robot, or to require the condition to persist — not to widen the
limit.

Do not infer the robot's physical state. Ask the operator, or measure it. The
two worst mistakes in this project's history were both inferences: concluding
low-level control was available because no locomotion process appeared in `ps`,
and recording that a run ended on a velocity abort when the operator had in
fact pressed the stop.
