#!/usr/bin/env bash
# Grant the G1's onboard Jetson Orin what the native 500 Hz loop needs in order
# to meet its deadlines, and verify the result.
#
# Two separate things are required, and both are lost over time:
#
#   1. The kernel is built with CONFIG_RT_GROUP_SCHED=y and user.slice is given
#      no real-time bandwidth, so nothing in a user session may run at a
#      real-time scheduling policy at all.  This grant resets to 0 on reboot.
#   2. The loop binary needs CAP_SYS_NICE to select SCHED_FIFO for itself and
#      CAP_IPC_LOCK to lock its pages, because the memlock limit (64 MB) is
#      smaller than the compiled model (95 MB).  File capabilities are lost
#      whenever the binary is rebuilt or replaced.
#
# Without these, the loop still runs, but it prints "SCHED_FIFO unavailable"
# and "mlockall unavailable" at start-up and is starved for tens of
# milliseconds at a time whenever the robot's own SLAM stack is busy.
#
# Usage:  SUDO_PASS=<robot password> bash orin_enable_realtime.sh
#
# The password is read from the environment and never written to the robot.
# To undo the bandwidth grant:  write 0 back to the cgroup file below.
# To undo the capabilities:     setcap -r on the binary.

set -euo pipefail

robot_binary=/home/unitree/bfm_teleop_fix5/build/g1_true23_bfm_lowcmd_loop
rt_file=/sys/fs/cgroup/cpu,cpuacct/user.slice/cpu.rt_runtime_us
rt_grant_us=200000
helper="$(dirname "$0")/fix5_onboard_ssh.sh"

if [ -z "${SUDO_PASS:-}" ]; then
  echo "SUDO_PASS is not set; refusing to continue" >&2
  exit 64
fi

remote=$(cat <<REMOTE
set -u
echo "real-time bandwidth before: \$(cat $rt_file)"
echo '$SUDO_PASS' | sudo -S -k -p "" sh -c "echo $rt_grant_us > $rt_file"
echo "real-time bandwidth after:  \$(cat $rt_file)"
echo '$SUDO_PASS' | sudo -S -k -p "" setcap cap_sys_nice,cap_ipc_lock+ep $robot_binary
echo "capabilities: \$(getcap $robot_binary)"
REMOTE
)

encoded=$(printf '%s' "$remote" | base64 -w0)
bash "$helper" run "echo $encoded | base64 -d > /tmp/orin_enable_realtime.sh; bash /tmp/orin_enable_realtime.sh; rm -f /tmp/orin_enable_realtime.sh"

cat <<'NOTE'

Re-run this after every reboot of the robot and after every rebuild or
redeployment of the loop binary.  The loop reports the problem itself: when
either grant is missing, "SCHED_FIFO unavailable" or "mlockall unavailable"
appears in the run's stdout.txt.
NOTE
