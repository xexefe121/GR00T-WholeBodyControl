param(
    [string]$RobotPassword = '123'
)

# Restore the robot-side deployment from this repository, in one command.
#
# Run it after any of these, all of which have happened:
#   - the robot's E-stop cut power, which reboots the Orin; an unclean shutdown
#     has deleted the freshly linked loop binary once;
#   - any rebuild of the loop binary, which silently drops its file
#     capabilities (cap_sys_nice, cap_ipc_lock);
#   - any reboot, which clears the user.slice real-time bandwidth grant;
#   - a change to the loop or tool sources in this repository.
#
# It copies the loop source and the diagnostic tool sources to the robot,
# builds both there against the vendored unitree_sdk2, restores capabilities
# and the real-time grant, and verifies the result.  It never arms and never
# commands the robot: nothing here publishes LowCmd.

$ErrorActionPreference = 'Stop'
$repo = 'Z:\codex\GR00T-WholeBodyControl-sonic-transfer-23dof'
$robot = '192.168.123.164'
$helper = '/mnt/z/codex/GR00T-WholeBodyControl-sonic-transfer-23dof/gear_sonic/scripts/fix5_onboard_ssh.sh'
$remote = '/home/unitree/bfm_teleop_fix5'
$loopSource = Join-Path $repo 'gear_sonic_deploy\src\g1\g1_deploy_onnx_ref\src\g1_true23_bfm_lowcmd_loop.cpp'
$toolDir = Join-Path $repo 'gear_sonic_deploy\src\g1\g1_deploy_onnx_ref\tools'
$tools = @('modeprobe', 'motorstate', 'tilt', 'waist', 'twitch')

function Invoke-Robot([string]$script) {
    # Remote logic always travels as base64: passing $ expressions through
    # wsl.exe from Windows loses them, and piping text converts line endings.
    $encoded = [Convert]::ToBase64String([Text.Encoding]::UTF8.GetBytes($script.Replace("`r`n", "`n")))
    $previous = $ErrorActionPreference
    $ErrorActionPreference = 'Continue'
    $result = wsl.exe -d Ubuntu-22.04 -- bash $helper run "echo $encoded | base64 -d > /tmp/rebuild_step.sh; bash /tmp/rebuild_step.sh; true"
    $ErrorActionPreference = $previous
    return $result
}

function Send-File([string]$local, [string]$remotePath) {
    # Large sources go through the helper's stdin; that path has been reliable
    # for the ~80 KB loop source, where a base64 argument exceeds the command
    # line limit.
    $previous = $ErrorActionPreference
    $ErrorActionPreference = 'Continue'
    Get-Content -Raw -LiteralPath $local | wsl.exe -d Ubuntu-22.04 -- bash $helper run "cat > $remotePath"
    $ErrorActionPreference = $previous
}

if (!(Test-Connection -ComputerName $robot -Count 2 -Quiet)) {
    throw "robot unreachable at $robot; power it on and check the Ethernet link"
}

Write-Host 'copying sources to the robot...'
Invoke-Robot "mkdir -p $remote/source $remote/tools" | Out-Null
Send-File $loopSource "$remote/source/g1_true23_bfm_lowcmd_loop.cpp"
foreach ($tool in $tools) { Send-File (Join-Path $toolDir "$tool.cpp") "$remote/tools/$tool.cpp" }

Write-Host 'building the loop and tools on the robot...'
$build = Invoke-Robot @"
set -u
R=$remote
cd "`$R"
# Line endings can arrive as CRLF from Windows; the compiler does not care, but
# be certain the build script itself is clean.
sed -i 's/\r`$//' fix7_onboard_build.sh
bash fix7_onboard_build.sh > /tmp/loop_build.log 2>&1
if grep -q 'error:' /tmp/loop_build.log; then
  echo 'LOOP BUILD FAILED'; grep -nE 'error:' /tmp/loop_build.log | head -8; exit 0
fi
test -x build/g1_true23_bfm_lowcmd_loop && echo 'loop=built' || echo 'loop=MISSING after build'
SDK=`$(find /home/unitree/g1_true23_onboard -type f -path '*/unitree_sdk2/CMakeLists.txt' -print -quit | xargs -r dirname)
for t in $($tools -join ' '); do
  g++ -O2 -std=c++17 -o "tools/`$t" "tools/`$t.cpp" \
    -I"`$SDK/include" -I"`$SDK/thirdparty/include" -I"`$SDK/thirdparty/include/ddscxx" \
    -I"`$SDK/thirdparty/include/iceoryx/v2.0.2" \
    "`$SDK/lib/aarch64/libunitree_sdk2.a" "`$SDK/thirdparty/lib/aarch64/libddscxx.so" \
    "`$SDK/thirdparty/lib/aarch64/libddsc.so" -lpthread > "/tmp/tool_`$t.log" 2>&1 \
    && echo "tool_`$t=built" || { echo "tool_`$t=FAILED"; head -4 "/tmp/tool_`$t.log"; }
done
"@
Write-Host ($build -join "`n")
if (($build -join ' ') -match 'FAILED|MISSING') { throw 'build failed on the robot; the errors are above' }

Write-Host 'restoring capabilities and the real-time grant...'
$grant = Invoke-Robot @"
B=$remote/build/g1_true23_bfm_lowcmd_loop
F=/sys/fs/cgroup/cpu,cpuacct/user.slice/cpu.rt_runtime_us
echo '$RobotPassword' | sudo -S -k -p "" setcap cap_sys_nice,cap_ipc_lock+ep "`$B"
echo '$RobotPassword' | sudo -S -k -p "" sh -c "echo 200000 > `$F"
echo "caps=`$(getcap "`$B")"
echo "rt_runtime=`$(cat `$F)"
echo "motion_mode: `$($remote/tools/modeprobe eth0 2>&1 | tail -1)"
"@
Write-Host ($grant -join "`n")
if (($grant -join ' ') -notmatch 'cap_sys_nice') { throw 'capabilities were not applied to the loop binary' }
if (($grant -join ' ') -notmatch 'rt_runtime=200000') { throw 'the real-time grant was not applied' }

Write-Host "`nrobot deployment restored and verified." -ForegroundColor Green
Write-Host 'Next: put the robot in damping, press L2+R2 for debug mode, then run run_g1_true23_live_teleop.ps1.'
