param(
    # Per-control step used by the policy stage only.  The qualified cap is
    # 0.100 rad; 0.035 sustained about a minute on a suspended robot before the
    # 6 rad/s measured-velocity abort fired legitimately.  Lower it for longer
    # runs, do not raise the velocity limit.
    [double]$PolicyBrakeStep = 0.035,
    [int]$PolicySeconds = 240,
    [string]$Clip = 'pico',
    [string]$RobotPassword = '123',
    [switch]$SkipRealtimeGrant,
    [string]$OutputRoot = 'E:\codex-artifacts\bfm_teleop_20260917\live_teleop'
)

# One command for a full live BFM teleop session on the G1.
#
# BEFORE running this, with the robot suspended in the gantry harness:
#   1. Put the robot into the damping state (the operator has done this with
#      the E-stop; L2+B on the remote is the documented route).
#   2. Press L2+R2 to enter debug mode.  Nothing actuates without this: the
#      control board accepts LowCmd and applies no torque until debug mode.
#   3. Keep a hand on the stop.  This script commands the real robot.
#
# It then does, in order: real-time grant, pre-flight, operator client, arm,
# the ladder up to the default pose, the BFM policy and clip stream, and the
# hand-over to policy control.  It stops everything cleanly on exit.

$ErrorActionPreference = 'Stop'
$repo = 'Z:\codex\GR00T-WholeBodyControl-sonic-transfer-23dof'
$robot = '192.168.123.164'
$helper = '/mnt/z/codex/GR00T-WholeBodyControl-sonic-transfer-23dof/gear_sonic/scripts/fix5_onboard_ssh.sh'
$scratch = Join-Path $env:TEMP 'g1_live_teleop'
New-Item -ItemType Directory -Force $scratch | Out-Null
$stamp = Get-Date -Format 'HHmmss'
$output = Join-Path $OutputRoot "session_$stamp"

function Invoke-Robot([string]$script) {
    $bytes = [Text.Encoding]::UTF8.GetBytes($script.Replace("`r`n", "`n"))
    $encoded = [Convert]::ToBase64String($bytes)
    $previous = $ErrorActionPreference
    $ErrorActionPreference = 'Continue'
    $result = wsl.exe -d Ubuntu-22.04 -- bash $helper run "echo $encoded | base64 -d > /tmp/live_step.sh; bash /tmp/live_step.sh; true"
    $ErrorActionPreference = $previous
    return $result
}

function Get-RobotStatus { (Invoke-Robot 'cat /dev/shm/g1_bringup_status.json 2>/dev/null') -join '' }

function Stop-Everything {
    Get-CimInstance Win32_Process -Filter "Name='python.exe'" -ErrorAction SilentlyContinue |
        Where-Object { $_.CommandLine -match 'bringup_operator|operator_driver|split_policy|teleop_sim' } |
        ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }
    Invoke-Robot 'pkill -f g1_true23_bfm_lowcmd_loop 2>/dev/null; sleep 1; echo "loops: $(pgrep -cf g1_true23_bfm_lowcmd_loop 2>/dev/null || echo 0)"' | Out-Null
}

try {
    if (!(Test-Connection -ComputerName $robot -Count 2 -Quiet)) {
        throw "robot unreachable at $robot; check the Ethernet link before anything else"
    }
    Write-Host "robot reachable" -ForegroundColor Green

    Stop-Everything

    if (!$SkipRealtimeGrant) {
        # Lost on every robot reboot.  Without it the 500 Hz loop is starved by
        # the robot's own SLAM stack and misses deadlines in bursts.
        $grant = Invoke-Robot @"
F=/sys/fs/cgroup/cpu,cpuacct/user.slice/cpu.rt_runtime_us
echo '$RobotPassword' | sudo -S -k -p "" sh -c "echo 200000 > `$F"
echo "rt_runtime=`$(cat `$F)  caps=`$(getcap /home/unitree/bfm_teleop_fix5/build/g1_true23_bfm_lowcmd_loop)"
"@
        Write-Host ($grant -join "`n")
    }

    # Pre-flight: the robot must be publishing state, and no motion mode may
    # hold the joints.  Torque is not checked here because a correct hold
    # command produces almost none; debug mode is what actually gates actuation.
    Write-Host ((Invoke-Robot '/tmp/modeprobe eth0 2>&1 | tail -1') -join "`n")

    # The operator client must be delivering heartbeats BEFORE the loop arms,
    # or the one-second deadman aborts within a second of arming.
    $commandFile = Join-Path $scratch 'operator_cmd.txt'
    Set-Content -Path $commandFile -Value '' -NoNewline
    $driver = Join-Path $scratch 'operator_driver.py'
    @"
import os, subprocess, sys, time
repo = r"$repo"
env = dict(os.environ); env["PYTHONPATH"] = repo
p = subprocess.Popen([sys.executable, "-m", "gear_sonic.scripts.run_g1_true23_bringup_operator",
                      "--endpoint", "tcp://${robot}:5912", "--heartbeat-seconds", "0.05"],
                     cwd=repo, env=env, stdin=subprocess.PIPE, text=True)
seen = 0
while p.poll() is None:
    try: lines = open(r"$commandFile").read().splitlines()
    except OSError: lines = []
    while seen < len(lines):
        command = lines[seen].strip(); seen += 1
        if command:
            p.stdin.write(command + "\n"); p.stdin.flush()
            print("sent:", command, flush=True)
    time.sleep(0.2)
"@ | Set-Content -Path $driver
    $operator = Start-Process -FilePath python -ArgumentList @($driver) -WorkingDirectory $repo -PassThru -WindowStyle Hidden `
        -RedirectStandardOutput (Join-Path $scratch 'operator.log') -RedirectStandardError (Join-Path $scratch 'operator.err')
    Start-Sleep -Seconds 4
    Write-Host "operator client up (pid $($operator.Id))" -ForegroundColor Green

    $arm = Invoke-Robot @"
TOKEN=`$(head -c 16 /dev/urandom | od -An -tx1 | tr -d ' \n')
echo "`$TOKEN" > /tmp/g1_operator_token; chmod 600 /tmp/g1_operator_token
: > /dev/shm/g1_bringup_status.json
R=/home/unitree/bfm_teleop_fix5
OUT=`$R/runs/live_$stamp; mkdir -p "`$OUT"
setsid nohup "`$R/build/g1_true23_bfm_lowcmd_loop" --source dds \
  --initial-command "`$R/source/initial_command.bin" \
  --model "`$R/thirdparty/g1_true23_bfm.mjb" \
  --state-endpoint tcp://${robot}:5910 --target-endpoint tcp://${robot}:5911 \
  --bringup-ladder --operator-endpoint tcp://${robot}:5912 \
  --hardware-bringup --arm --dds-domain 0 --dds-interface eth0 \
  --operator-token-file /tmp/g1_operator_token --operator-token "`$TOKEN" \
  --policy-brake-step $PolicyBrakeStep \
  --output "`$OUT" --ticks 900000 > "`$OUT/stdout.txt" 2>&1 < /dev/null &
sleep 8
head -3 "`$OUT/stdout.txt"
cat /dev/shm/g1_bringup_status.json
"@
    Write-Host ($arm -join "`n")
    if (($arm -join ' ') -match 'refused') { throw 'arming refused; read the reason above and correct it before retrying' }

    foreach ($stage in @('zero torque', 'damping', 'position hold', 'default pose')) {
        Add-Content -Path $commandFile -Value 'advance'
        Start-Sleep -Seconds 6
        Write-Host ("-> {0}: {1}" -f $stage, (Get-RobotStatus))
        if ((Get-RobotStatus) -match 'aborted') { throw "aborted while advancing to $stage" }
    }

    # BFM and the clip stream.  The barrier matters: started without it the
    # publisher streams the whole clip while the policy is still building its
    # model, the admission gate never sees a packet, and the policy correctly
    # emits nothing.
    $ready = Join-Path $scratch 'policy_ready.json'
    $publisherReady = Join-Path $scratch 'publisher_ready.json'
    $release = Join-Path $scratch 'release_clock'
    foreach ($file in @($ready, $publisherReady, $release)) { Remove-Item $file -ErrorAction SilentlyContinue }
    $policy = Start-Process -FilePath python -WorkingDirectory $repo -PassThru -WindowStyle Hidden `
        -RedirectStandardOutput (Join-Path $scratch 'policy.log') -RedirectStandardError (Join-Path $scratch 'policy.err') -ArgumentList @(
        '-m', 'gear_sonic.scripts.run_g1_true23_bfm_split_policy', 'run', '--placement', 'windows',
        '--state-endpoint', "tcp://${robot}:5910", '--target-endpoint', "tcp://${robot}:5911",
        '--teleop-endpoint', 'tcp://127.0.0.1:6070', '--output', $output,
        '--duration-seconds', $PolicySeconds, '--priority', 'high', '--torch-threads', '4',
        '--pacer-spin-ms', '1.0', '--ready-file', $ready)
    $publisher = Start-Process -FilePath python -WorkingDirectory $repo -PassThru -WindowStyle Hidden `
        -RedirectStandardOutput (Join-Path $scratch 'publisher.log') -RedirectStandardError (Join-Path $scratch 'publisher.err') -ArgumentList @(
        '-m', 'gear_sonic.scripts.run_g1_true23_bfm_teleop_sim', 'publish', '--clip', $Clip,
        '--endpoint', 'tcp://127.0.0.1:6070', '--start-delay', '0',
        '--ready-file', $publisherReady, '--start-file', $release,
        '--data-root', 'C:\Users\camer\sonic23_sim_artifacts\internet_pico_20260909_v1')
    Write-Host "policy building its model, this takes a minute..."
    $deadline = (Get-Date).AddSeconds(180)
    while (((Get-Date) -lt $deadline) -and (-not ((Test-Path $ready) -and (Test-Path $publisherReady)))) { Start-Sleep -Milliseconds 500 }
    if (-not ((Test-Path $ready) -and (Test-Path $publisherReady))) { throw 'policy or publisher never signalled ready' }
    Start-Sleep -Milliseconds 500
    New-Item -ItemType File -Path $release | Out-Null
    Start-Sleep -Seconds 10
    Write-Host ("targets flowing: {0}" -f (Get-RobotStatus))

    Add-Content -Path $commandFile -Value 'advance'
    Write-Host "`n=== POLICY: BFM is now driving the robot ===" -ForegroundColor Cyan
    $until = (Get-Date).AddSeconds($PolicySeconds)
    while ((Get-Date) -lt $until) {
        Start-Sleep -Seconds 10
        $status = Get-RobotStatus
        Write-Host $status
        if ($status -match 'aborted') {
            Write-Host "`nstopped by an abort; the reason is in the line above" -ForegroundColor Yellow
            break
        }
    }
}
finally {
    Write-Host "`nstopping everything and leaving the robot limp..."
    Stop-Everything
    Write-Host ((Invoke-Robot '/tmp/modeprobe eth0 2>&1 | tail -1') -join "`n")
    Write-Host "session evidence: $output"
}
