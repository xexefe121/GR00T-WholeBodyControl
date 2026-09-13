# Replay the qualified WALK002 HYBRID commands through native physics and compare every step.
# This is recorded-command replay, not a live teleoperation controller.
param(
    [ValidatePattern('^[A-Za-z0-9_-]+$')]
    [string]$RunName = ('walk002_recorded_replay_' + (Get-Date -Format 'yyyyMMdd_HHmmss'))
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
$repoWin = 'Z:\codex\GR00T-WholeBodyControl-sonic-transfer-23dof'
$repoWsl = '/mnt/z/codex/GR00T-WholeBodyControl-sonic-transfer-23dof'
$baseWin = 'E:\codex-artifacts\sonic23_teleop_resume_20260911'
$baseWsl = '/mnt/e/codex-artifacts/sonic23_teleop_resume_20260911'
$outputWin = Join-Path $baseWin $RunName
$outputWsl = "$baseWsl/$RunName"
if (Test-Path -LiteralPath $outputWin) { throw 'Choose a new RunName; existing evidence is never overwritten.' }

function Read-Sha256([string]$Path) {
    $stream = [IO.File]::OpenRead($Path)
    $hasher = [Security.Cryptography.SHA256]::Create()
    try { return ([BitConverter]::ToString($hasher.ComputeHash($stream))).Replace('-', '').ToLowerInvariant() }
    finally { $stream.Dispose(); $hasher.Dispose() }
}

function Assert-Pin([string]$Path, [string]$Expected) {
    if ((Read-Sha256 $Path) -ne $Expected) { throw "Pinned input changed: $Path" }
}

$qualificationPath = Join-Path $baseWin 'walk002_hybrid_root_qualification_v1/qualification.json'
Assert-Pin $qualificationPath 'a7eaec31dd572724921e2d2986ab8c6ab5db33008f6b3c1af5d2dbd72bf92e4b'
Assert-Pin (Join-Path $baseWin 'walk002_qualified_package_v1/replay_input_pins.json') '9fbefe58ce5c4ae87f2d7dbfd62efe5e222f4ee1efc1c5c984c1a45e5d9bd02f'
$inputPins = Get-Content -Raw -LiteralPath (Join-Path $baseWin 'walk002_qualified_package_v1/replay_input_pins.json') | ConvertFrom-Json
foreach ($entry in $inputPins.PSObject.Properties) { Assert-Pin $entry.Name $entry.Value }
$qualification = Get-Content -Raw -LiteralPath $qualificationPath | ConvertFrom-Json
if (($qualification.complete_offline_walk002_hybrid_pass -ne $true) -or ($qualification.both_quiet_windows_pass -ne $true)) {
    throw 'The bound full lifecycle and separate hold must already be qualified.'
}
foreach ($entry in $qualification.traces.PSObject.Properties) {
    Assert-Pin $entry.Value.path $entry.Value.sha256
}
foreach ($entry in $qualification.independent_reports.PSObject.Properties) {
    Assert-Pin $entry.Value.path $entry.Value.sha256
}
New-Item -ItemType Directory -Path $outputWin | Out-Null

function Invoke-RecordedReplay([string]$Part, [string]$Producer, [string]$Fixture, [int]$Controls) {
    $arguments = @(
        '-d', 'Ubuntu-22.04', '--cd', $repoWsl,
        '--', 'bash', "$repoWsl/artifacts/teleop_six_hour_20260910/RUN_PASSING_WALK_WSL.sh",
        'env', 'OMP_NUM_THREADS=1', 'OPENBLAS_NUM_THREADS=1', 'MKL_NUM_THREADS=1',
        '/mnt/e/codex_sonic_runtime/mjbatch323_20260910/venv/bin/python',
        "$baseWsl/walk002_qualified_package_v1/replay/audit_restored_native_segment.py",
        '--frozen-repo', "$baseWsl/walk002_qualified_package_v1/replay/repo",
        '--oracle', "$baseWsl/walk002_qualified_package_v1/replay/g1_true23_feasibility_referee.py",
        '--bundle', 'artifacts/teleop_six_hour_20260910/mjbatch_native23_inputs_v1',
        '--fixture', "$baseWsl/$Fixture/initial_integration_state.npz",
        '--trace', "$baseWsl/$Producer/trace.npz", '--clip', 'walk002',
        '--requested-controls', $Controls.ToString(), '--output', "$outputWsl/$Part"
    )
    & wsl.exe @arguments 1> (Join-Path $outputWin "$Part.stdout.log") 2> (Join-Path $outputWin "$Part.stderr.log")
    if ($LASTEXITCODE -ne 0) { throw "Recorded physics replay failed; see $outputWin/$Part.stdout.log" }
    $report = Get-Content -Raw -LiteralPath (Join-Path $outputWin "$Part/report.json") | ConvertFrom-Json
    if (($report.independent_segment_pass -ne $true) -or ($report.compared_physics_steps -ne (10 * $Controls))) {
        throw "Incomplete physical replay: $Part"
    }
    foreach ($check in $report.original_trace_comparison.PSObject.Properties) {
        if ($check.Value -ne $true) { throw "Recorded sample mismatch: $Part / $($check.Name)" }
    }
}

Invoke-RecordedReplay 'lifecycle' 'walk002_terminal_bfm_hybrid_v1' 'walk002_canonical_initial_fixture_v1' 1417
Invoke-RecordedReplay 'separate_hold' 'walk002_terminal_bfm_hybrid_v1/post_lifecycle_hold_5s' 'walk002_hybrid_hold_independent_fixture_v1' 250
$result = [ordered]@{
    kind = 'qualified_walk002_hybrid_recorded_command_native_physics_replay'
    qualification_sha256 = Read-Sha256 $qualificationPath
    lifecycle_physics_steps = 14170
    separate_hold_physics_steps = 2500
    every_recorded_sample_bitexact = $true
    new_controller_inference = $false
    live_teleoperation_qualified = $false
    completed_utc = [DateTime]::UtcNow.ToString('o')
}
$result | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $outputWin 'replay_result.json')
Write-Output 'PASS: all 16,670 recorded native physics steps match exactly.'
Write-Output "Evidence: $outputWin"
Write-Output 'Recorded-command replay only; live teleoperation remains unqualified.'
