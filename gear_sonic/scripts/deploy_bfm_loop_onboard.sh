#!/usr/bin/env bash
# Build the read-only Fix 5 timing loop in a new directory on the onboard PC.
# The existing local SSH credential is obtained only in memory from the prior
# read-only inspection helper; it is never written or printed.
set -euo pipefail

robot=192.168.123.164
local_address=$(ip -4 -o addr show dev eth0 | awk '{print $4}' | cut -d/ -f1 | grep '^192\.168\.123\.' | head -1)
if [ -z "$local_address" ]; then echo "No 192.168.123.x address on eth0; connect the robot Ethernet cable. No network configuration changed." >&2; exit 69; fi
remote_root=/home/unitree/bfm_teleop_fix5
protected_root=/home/unitree/g1_true23_onboard
repo=/mnt/z/codex/GR00T-WholeBodyControl-sonic-transfer-23dof
inspection_helper="$repo/artifacts/onboard_inspection_20260912/run_readonly_ssh.py"
replay=/mnt/e/codex-artifacts/bfm_teleop_20260917/fix4/pico_lowstate_500hz.flatbin
initial=/mnt/e/codex-artifacts/bfm_teleop_20260917/fix4/initial_command.bin

if ! ping -c 1 -W 1 "$robot" >/dev/null 2>&1; then
  echo "robot not reachable at $robot; cable and power the G1 before deployment" >&2
  exit 69
fi
if [[ ! -r "$inspection_helper" || ! -r "$replay" || ! -r "$initial" ]]; then
  echo "required local Fix 5 source, inspection helper, or exported replay is missing" >&2
  exit 66
fi
password=$(python3 - "$inspection_helper" <<'PY'
import ast, pathlib, sys
tree = ast.parse(pathlib.Path(sys.argv[1]).read_text())
for node in ast.walk(tree):
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == 'dict':
        for keyword in node.keywords:
            if keyword.arg == 'SSHPASS' and isinstance(keyword.value, ast.Constant):
                print(keyword.value.value, end='')
                raise SystemExit
raise SystemExit('existing inspection authentication method was not found')
PY
)
export SSHPASS="$password"
unset password
ssh_base=(sshpass -e ssh -b "$local_address" -o ConnectTimeout=5 -o StrictHostKeyChecking=yes -o NumberOfPasswordPrompts=1 "unitree@$robot")
scp_base=(sshpass -e scp -o ConnectTimeout=5 -o StrictHostKeyChecking=yes -o NumberOfPasswordPrompts=1)

"${ssh_base[@]}" "test '$remote_root' != '$protected_root' && test ! -e '$remote_root' && test -d '$protected_root' || { echo 'refusing overwrite or protected checkout missing' >&2; exit 65; }; mkdir -p '$remote_root/source' '$remote_root/replay'"
"${scp_base[@]}" -3 -o BindAddress="$local_address" \
  "$repo/gear_sonic_deploy/src/g1/g1_deploy_onnx_ref/src/g1_true23_bfm_lowcmd_loop.cpp" \
  "$repo/gear_sonic/scripts/fix5_onboard_CMakeLists.txt" \
  "$replay" "$initial" "unitree@$robot:$remote_root/source/"
"${ssh_base[@]}" "mv '$remote_root/source/fix5_onboard_CMakeLists.txt' '$remote_root/source/CMakeLists.txt'; sdk=\$(find '$protected_root' -type f -path '*/unitree_sdk2/CMakeLists.txt' -print -quit | xargs -r dirname); test -n \"\$sdk\" || { echo 'unitree_sdk2 not found in protected checkout' >&2; exit 67; }; cmake -S '$remote_root/source' -B '$remote_root/build' -DUNITREE_SDK_ROOT=\"\$sdk\"; cmake --build '$remote_root/build' --target g1_true23_bfm_lowcmd_loop -j2; binary='$remote_root/build/g1_true23_bfm_lowcmd_loop'; test -x \"\$binary\"; printf 'built binary: %s\\n' \"\$binary\"; file \"\$binary\""
unset SSHPASS
echo "To remove only this Fix 5 build: sshpass -e ssh -b $local_address unitree@$robot 'rm -rf $remote_root'"
