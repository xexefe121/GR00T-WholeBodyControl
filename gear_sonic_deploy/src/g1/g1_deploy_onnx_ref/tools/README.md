# G1 on-robot diagnostic tools

Small standalone programs built on the robot's Jetson Orin against the vendored
unitree_sdk2. They answer, in seconds, the questions that cost the most time
during bring-up. `gear_sonic/scripts/g1_true23_robot_rebuild.ps1` copies them to
`/home/unitree/bfm_teleop_fix5/tools/` on the robot and builds them there.
Keep them in that persistent directory: `/tmp` is wiped on every reboot, and
the robot reboots whenever its E-stop cuts power.

| tool | question it answers | commands the robot? |
|---|---|---|
| `modeprobe eth0` | Does Unitree's motion-control service still hold the joints? | no |
| `motorstate eth0` | Per-joint mode, position, velocity and measured torque | no |
| `tilt eth0` | Pelvis tilt from upright, against the 20 degree abort limit | no |
| `waist eth0` | Live waist-yaw readout while straightening the torso by hand | no |
| `twitch eth0 <slot> <offset> <kp>` | Is the robot actually actuating? | **yes**: one joint, briefly |

`twitch` is the only one that publishes LowCmd. It commands the measured pose
plus a small offset on one joint and reports how far the joint moved. It exists
because position hold cannot tell "acting on our commands" from "ignoring them":
hold commands the pose the robot is already in, so both produce almost no
torque. Without debug mode the robot accepts every command and applies nothing,
and `twitch` reports roughly zero movement. A few hundredths of a radian or more
means actuation is live.
