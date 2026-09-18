param(
    [string]$Output = 'E:\codex-artifacts\bfm_teleop_20260917\fix5',
    [int]$RunCount = 2,
    [int]$StatePort = 5560,
    [int]$TargetPort = 5561,
    [int]$TeleopPort = 5562
)
# Fix 5 qualification: policy remains on Windows; the loop runs on the G1.
# The loop binary is hard-wired to the domain-232 loopback test publisher.
$ErrorActionPreference = 'Stop'
$repo = 'Z:\codex\GR00T-WholeBodyControl-sonic-transfer-23dof'
$robot = '192.168.123.164'
$remoteRoot = '/home/unitree/bfm_teleop_fix5'
$sshHelper = '/mnt/z/codex/GR00T-WholeBodyControl-sonic-transfer-23dof/gear_sonic/scripts/fix5_onboard_ssh.sh'
$binary = "$remoteRoot/build/g1_true23_bfm_lowcmd_loop"
$replay = "$remoteRoot/source/pico_lowstate_500hz.flatbin"
$initial = "$remoteRoot/source/initial_command.bin"
if ($RunCount -lt 1) { throw 'RunCount must be positive.' }

$ping = [System.Net.NetworkInformation.Ping]::new()
try { $reachable = $ping.Send($robot, 1000).Status -eq [System.Net.NetworkInformation.IPStatus]::Success } catch { $reachable = $false }
if (!$reachable) {
    [Console]::Error.WriteLine("robot not reachable at $robot; cable and power the G1 before timing qualification")
    exit 69
}
if (Test-Path -LiteralPath $Output) { throw "output already exists: $Output" }
New-Item -ItemType Directory -Path $Output | Out-Null

function ConvertTo-WslPath([string]$windowsPath) {
    # wsl.exe strips backslashes from arguments, so convert locally: E: -> /mnt/e/a/b
    $full = [System.IO.Path]::GetFullPath($windowsPath)
    return '/mnt/' + $full.Substring(0,1).ToLower() + ($full.Substring(2).Replace('\', '/'))
}

function Start-RemoteLoop([string]$remoteCommand, [string]$directory) {
    Start-Process -FilePath wsl.exe -ArgumentList @('-d','Ubuntu-22.04','--','bash',$sshHelper,'run',$remoteCommand) `
        -PassThru -RedirectStandardOutput (Join-Path $directory 'remote_loop.log') `
        -RedirectStandardError (Join-Path $directory 'remote_loop.err')
}

$runs = @()
for ($index = 1; $index -le $RunCount; $index++) {
    $name = ('run_{0:D2}' -f $index)
    $run = Join-Path $Output $name
    $native = Join-Path $run 'native'
    $policyDir = Join-Path $run 'policy'
    New-Item -ItemType Directory -Path $native | Out-Null
    $remoteRun = "$remoteRoot/runs/$name"
    $runStatePort = $StatePort + 10 * $index
    $runTargetPort = $TargetPort + 10 * $index
    $runTeleopPort = $TeleopPort + 10 * $index
    $state = "tcp://${robot}:$runStatePort"
    $target = "tcp://${robot}:$runTargetPort"
    $policy = Start-Process -FilePath python -ArgumentList @(
        '-m','gear_sonic.scripts.run_g1_true23_bfm_split_policy','run',
        '--placement','windows','--state-endpoint',$state,'--target-endpoint',$target,
        '--teleop-endpoint',"tcp://127.0.0.1:$runTeleopPort",'--output',$policyDir,
        '--duration-seconds','115.6','--priority','high','--torch-threads','4'
    ) -WorkingDirectory $repo -PassThru -RedirectStandardOutput (Join-Path $run 'policy.log') -RedirectStandardError (Join-Path $run 'policy.err')
    $publisher = Start-Process -FilePath python -ArgumentList @(
        '-m','gear_sonic.scripts.run_g1_true23_bfm_teleop_sim','publish',
        '--clip','pico','--endpoint',"tcp://127.0.0.1:$runTeleopPort",'--start-delay','12',
        '--data-root','C:\Users\camer\sonic23_sim_artifacts\internet_pico_20260909_v1'
    ) -WorkingDirectory $repo -PassThru -RedirectStandardOutput (Join-Path $run 'publisher.log') -RedirectStandardError (Join-Path $run 'publisher.err')
    # Model construction is outside the measured 50 Hz region.  Start the loop
    # only after the policy has had its established preparation window.
    Start-Sleep -Seconds 12
    $remoteCommand = "bash $remoteRoot/run_loop.sh $name 57800 $runStatePort $runTargetPort"
    $loop = Start-RemoteLoop $remoteCommand $run
    try { $loop.WaitForExit(); $policy.WaitForExit(); $publisher.WaitForExit() } finally {
        foreach ($process in @($loop,$policy,$publisher)) { if (!$process.HasExited) { Stop-Process -Id $process.Id -Force } }
    }
    foreach ($file in @('loop_report.json','loop_ticks.csv')) {
        $destination = (ConvertTo-WslPath (Join-Path $native $file))
        wsl.exe -d Ubuntu-22.04 -- bash $sshHelper copy "$remoteRun/$file" $destination
    }
    $loop.Refresh(); $policy.Refresh(); $publisher.Refresh()
    $status = [ordered]@{ loop_exit=$loop.ExitCode; policy_exit=$policy.ExitCode; publisher_exit=$publisher.ExitCode; remote_run=$remoteRun }
    $status | ConvertTo-Json | Set-Content -NoNewline (Join-Path $run 'process_status.json')
    $loopReport = Get-Content -Raw (Join-Path $native 'loop_report.json') | ConvertFrom-Json
    $policyReport = Get-Content -Raw (Join-Path $policyDir 'policy_report.json') | ConvertFrom-Json
    $summary = [ordered]@{
        run=$name; loop_exit=$loop.ExitCode; policy_exit=$policy.ExitCode; publisher_exit=$publisher.ExitCode
        targets_received=$loopReport.targets_received; valid_exchange=($loopReport.targets_received -ge 5700)
        loop_500hz=[ordered]@{misses=$loopReport.deadline_misses; start_lateness_ms=$loopReport.start_lateness_ms; work_ms=$loopReport.work_ms; lowcmd_gap_ms=$loopReport.lowcmd_gap_ms}
        policy_50hz=[ordered]@{misses=$policyReport.path_50hz.deadline_misses; start_lateness_ms=$policyReport.path_50hz.start_lateness_ms; work_ms=$policyReport.path_50hz.work_ms}
        ethernet_ipc_round_trip_ms=$loopReport.ipc_round_trip_ms; target_age_ms=$loopReport.target_age_ms
    }
    $summary | ConvertTo-Json -Depth 8 | Set-Content -NoNewline (Join-Path $run 'summary.json')
    $runs += $summary
}
$pass = @($runs | Where-Object { $_.valid_exchange -and $_.loop_500hz.misses -eq 0 -and $_.policy_50hz.misses -eq 0 -and $_.loop_500hz.lowcmd_gap_ms.max -lt 4.0 }).Count -eq $RunCount
$overall = [ordered]@{ runs=$runs; required_consecutive_runs=$RunCount; passed=$pass; criterion='each run: at least 5700 targets received by the robot loop, zero 500 Hz misses, zero 50 Hz misses, lowcmd gap max under 4 ms' }
$overall | ConvertTo-Json -Depth 9 | Set-Content -NoNewline (Join-Path $Output 'summary.json')
if (!$pass) { exit 2 }
