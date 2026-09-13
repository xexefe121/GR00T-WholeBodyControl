param(
    [ValidatePattern('^[A-Za-z0-9_-]+$')]
    [string]$RunName = ('motion_prior_ppo_' + (Get-Date -Format 'yyyyMMdd_HHmmss'))
)
$ErrorActionPreference = 'Stop'
$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
$artifactRoot = 'E:\codex-artifacts\sonic23_teleop_resume_20260911\onboard_factory_firmware_v1'
$outputPath = Join-Path $artifactRoot $RunName
if (Test-Path -LiteralPath $outputPath) { throw "Output already exists: $outputPath" }
foreach ($relative in @('motion_prior_data_v1\transitions.npz', 'native_mjbatch_lifecycle_ppo_v1\actor_00100.pt')) {
    if (-not (Test-Path -LiteralPath (Join-Path $artifactRoot $relative))) { throw "Missing prepared input: $relative" }
}
$linuxBase = '/mnt/e/codex-artifacts/sonic23_teleop_resume_20260911'
$linuxFirmware = "$linuxBase/onboard_factory_firmware_v1"
$trainArguments = @(
    '-u', '-m', 'gear_sonic.scripts.train_g1_true23_factory_dynamics',
    '--firmware', $linuxFirmware,
    '--bank', "$linuxBase/causal_dynamics_v1/focused_walk002_task_closure_bank_v1",
    '--output', "$linuxFirmware/$RunName",
    '--locomotion-base', '--task-commands', '--command-space-exploration', '--full-body-corrections',
    '--task-base-checkpoint', "$linuxFirmware/native_mjbatch_lifecycle_ppo_v1/actor_00100.pt",
    '--motion-prior', "$linuxFirmware/motion_prior_data_v1/transitions.npz",
    '--num-envs', '512', '--steps', '64', '--updates', '200', '--wall-minutes', '40',
    '--bootstrap-updates', '0', '--imitation-weight', '0', '--canonical-starts', '--canonical-worlds', '4',
    '--command-delay-substeps', '2', '--continue-tracking-errors', '--decoupled-critic',
    '--critic-warmup-updates', '20', '--checkpoint-updates', '100', '200'
)
Write-Host "One bounded local GPU pilot. Results: $outputPath"
& wsl.exe --cd $projectRoot --exec bash -c 'mountpoint -q /mnt/e || mount -t drvfs E: /mnt/e; exec "$@"' 'motion-prior-pilot' '/root/.venvs/g1_true23_mjlab/bin/python' @trainArguments
if ($LASTEXITCODE -ne 0) { throw "Pilot exited with code $LASTEXITCODE. Inspect $outputPath" }
