param(
    [string]$OutputRoot = 'E:\codex-artifacts\bfm_teleop_20260917\fix8_windows_scheduling_stalls'
)

$ErrorActionPreference = 'Stop'
$runner = 'Z:\codex\GR00T-WholeBodyControl-sonic-transfer-23dof\gear_sonic\scripts\run_fix8_windows_replay_sweep.ps1'
$baseline = Join-Path $OutputRoot 'baseline_default_v2\summary.json'
$deadline = [DateTime]::UtcNow.AddHours(1)
while (!(Test-Path -LiteralPath $baseline)) {
    if ([DateTime]::UtcNow -gt $deadline) { throw 'baseline did not complete successfully; candidate sweep was not started' }
    Start-Sleep -Seconds 10
}

$configs = @(
    @{name='timer_0_5ms'; port=6010; args=@('-TimerResolution05ms')},
    @{name='torch_threads_1'; port=6110; args=@('-TorchThreads','1')},
    @{name='torch_threads_2'; port=6210; args=@('-TorchThreads','2')},
    @{name='omp_wait_passive'; port=6310; args=@('-OmpWaitPolicy','passive')},
    @{name='realtime_priority'; port=6410; args=@('-Priority','realtime')},
    @{name='mmcss_pro_audio'; port=6510; args=@('-Priority','mmcss')},
    @{name='affinity_off_cpu0'; port=6610; args=@('-Affinity','role-separated')},
    @{name='memory_residency'; port=6710; args=@('-MemoryResidency')}
)
foreach ($config in $configs) {
    & $runner -ConfigName $config.name -RunCount 3 -OutputRoot $OutputRoot -BasePort $config.port @($config.args)
    if ($LASTEXITCODE -ne 0) { throw "candidate failed: $($config.name)" }
}
