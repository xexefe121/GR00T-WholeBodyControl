param(
    [Parameter(Mandatory=$true)][string]$ConfigName,
    [int]$RunCount = 3,
    [string]$OutputRoot = 'E:\codex-artifacts\bfm_teleop_20260917\fix8_windows_scheduling_stalls',
    [int]$BasePort = 5710,
    [int]$TorchThreads = 4,
    [ValidateSet('default','passive')][string]$OmpWaitPolicy = 'default',
    [ValidateSet('high','realtime','mmcss')][string]$Priority = 'high',
    [ValidateSet('none','role-separated')][string]$Affinity = 'none',
    [switch]$TimerResolution05ms,
    [switch]$MemoryResidency,
    [double]$PacerSpinMs = 0.0,
    [switch]$NoTrace  # the per-control trace is itself first-call expensive; the qualification runs without it
)

# Localhost-only Fix 8 timing runner.  It has no DDS import, robot address, or
# robot-facing socket; the state replay drains targets only to prevent IPC backpressure.
$ErrorActionPreference = 'Stop'
$repo = 'Z:\codex\GR00T-WholeBodyControl-sonic-transfer-23dof'
$sensor = 'E:\codex-artifacts\bfm_teleop_20260917\fix4\pico_lowstate_500hz.flatbin'
$snapshots = 'E:\codex-artifacts\bfm_teleop_20260917\fix7\estimator_snapshots_cpp_retry.csv'
$dataRoot = 'C:\Users\camer\sonic23_sim_artifacts\internet_pico_20260909_v1'
$out = Join-Path $OutputRoot $ConfigName
foreach ($path in @($repo, $sensor, $snapshots, $dataRoot)) {
    if (!(Test-Path -LiteralPath $path)) { throw "required Fix 8 input is absent: $path" }
}
if (Test-Path -LiteralPath $out) { throw "output exists: $out" }
New-Item -ItemType Directory -Path $out | Out-Null

$runs = @()
for ($index = 1; $index -le $RunCount; $index++) {
    $run = Join-Path $out ('run_{0:D2}' -f $index)
    New-Item -ItemType Directory -Path $run | Out-Null
    $offset = 10 * ($index - 1)
    $state = "tcp://127.0.0.1:$($BasePort + $offset)"
    $target = "tcp://127.0.0.1:$($BasePort + $offset + 1)"
    $teleop = "tcp://127.0.0.1:$($BasePort + $offset + 2)"
    $policyReady = Join-Path $run 'policy_initialized.json'
    $publisherReady = Join-Path $run 'publisher_bound.json'
    $start = Join-Path $run 'release_source_clock'
    $policyArgs = @('-m','gear_sonic.scripts.run_g1_true23_bfm_split_policy','run',
        '--placement','windows','--state-endpoint',$state,'--target-endpoint',$target,
        '--teleop-endpoint',$teleop,'--output',(Join-Path $run 'policy'),
        '--duration-seconds','115.6','--priority',$Priority,'--torch-threads',$TorchThreads,
        '--omp-wait-policy',$OmpWaitPolicy,'--affinity',$Affinity,'--ready-file',$policyReady)
    if (!$NoTrace) { $policyArgs += '--trace-windows' }
    if ($PacerSpinMs -gt 0) { $policyArgs += @('--pacer-spin-ms',$PacerSpinMs) }
    if ($TimerResolution05ms) { $policyArgs += '--timer-resolution-0-5ms' }
    if ($MemoryResidency) { $policyArgs += '--memory-residency' }
    $publisherArgs = @('-m','gear_sonic.scripts.run_g1_true23_bfm_teleop_sim','publish',
        '--clip','pico','--endpoint',$teleop,'--start-delay','0','--ready-file',$publisherReady,
        '--start-file',$start,'--data-root',$dataRoot)
    $policy = Start-Process -FilePath python -ArgumentList $policyArgs -WorkingDirectory $repo -PassThru `
        -RedirectStandardOutput (Join-Path $run 'policy.log') -RedirectStandardError (Join-Path $run 'policy.err')
    $publisher = Start-Process -FilePath python -ArgumentList $publisherArgs -WorkingDirectory $repo -PassThru `
        -RedirectStandardOutput (Join-Path $run 'publisher.log') -RedirectStandardError (Join-Path $run 'publisher.err')
    $replay = $null
    $replayExit = $policyExit = $publisherExit = $null
    try {
        $deadline = [DateTime]::UtcNow.AddSeconds(90)
        while (!(Test-Path -LiteralPath $policyReady) -or !(Test-Path -LiteralPath $publisherReady)) {
            if ([DateTime]::UtcNow -ge $deadline) { throw 'Fix 8 readiness barrier timed out' }
            Start-Sleep -Milliseconds 25
        }
        Start-Sleep -Milliseconds 500 # established PUB/SUB subscription handshake
        New-Item -ItemType File -Path $start | Out-Null
        $replayArgs = @('-m','gear_sonic.scripts.run_g1_true23_bfm_split_policy','local-replay',
            '--sensor-stream',$sensor,'--native-snapshots',$snapshots,'--state-endpoint',$state,
            '--target-endpoint',$target,'--output',(Join-Path $run 'replay'),'--duration-seconds','115.6')
        $replay = Start-Process -FilePath python -ArgumentList $replayArgs -WorkingDirectory $repo -PassThru `
            -RedirectStandardOutput (Join-Path $run 'replay.log') -RedirectStandardError (Join-Path $run 'replay.err')
        $replay.WaitForExit(); $replayExit = $replay.ExitCode
        $policy.WaitForExit(); $policyExit = $policy.ExitCode
        $publisher.WaitForExit(); $publisherExit = $publisher.ExitCode
    } finally {
        foreach ($process in @($replay, $policy, $publisher)) {
            if ($null -ne $process) {
                $process.Refresh()
                if (!$process.HasExited) { Stop-Process -Id $process.Id -Force }
            }
        }
    }
    $reportsPresent = (Test-Path -LiteralPath (Join-Path $run 'replay\replay_report.json')) -and (Test-Path -LiteralPath (Join-Path $run 'policy\policy_report.json'))
    $status = [ordered]@{config=$ConfigName; run=$index; replay_exit=$replayExit; policy_exit=$policyExit; publisher_exit=$publisherExit; reports_present=$reportsPresent}
    $status | ConvertTo-Json | Set-Content -NoNewline (Join-Path $run 'process_status.json')
    if (!$reportsPresent) { throw "Fix 8 reports absent after child completion: $($status | ConvertTo-Json -Compress)" }
    $report = Get-Content -Raw (Join-Path $run 'policy\policy_report.json') | ConvertFrom-Json
    $p = $report.path_50hz
    $runs += [ordered]@{run=$index; deadline_misses=$p.deadline_misses; work_ms=$p.work_ms; start_lateness_ms=$p.start_lateness_ms; duty_cycle_percent=$p.duty_cycle_percent; outliers=@($p.outlier_activity).Count}
}
$summary = [ordered]@{config=$ConfigName; run_count=$RunCount; configuration=[ordered]@{torch_threads=$TorchThreads;omp_wait_policy=$OmpWaitPolicy;priority=$Priority;affinity=$Affinity;timer_resolution_0_5ms=[bool]$TimerResolution05ms;memory_residency=[bool]$MemoryResidency};runs=$runs}
$summary | ConvertTo-Json -Depth 8 | Set-Content -NoNewline (Join-Path $out 'summary.json')
