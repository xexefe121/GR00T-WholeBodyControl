# Replay the qualified PICO commands through native physics and compare every step.
# This is recorded-command replay, not a live teleoperation controller.
param(
    [ValidatePattern('^[A-Za-z0-9_-]+$')]
    [string]$RunName = ('pico_recorded_replay_' + (Get-Date -Format 'yyyyMMdd_HHmmss'))
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

$qualificationPath = Join-Path $baseWin 'pico_full_root_qualification_v1/qualification.json'
Assert-Pin $qualificationPath '746502265523f527a7836f8b091ba521a56078629b1943468eff04f8bfa61daa'
Assert-Pin (Join-Path $repoWin 'artifacts/teleop_resume_20260911/audit_restored_native_segment.py') 'd9ff4da640b6814e5d6f1aa5a550230b53ca202e8be6a2f7679bdef1aa715a4e'
Assert-Pin (Join-Path $repoWin 'gear_sonic/utils/g1_true23_feasibility_referee.py') '0027210cda5a44255debecd7151ecd454e2281a5ff4e3137641d102701fb1330'
Assert-Pin (Join-Path $baseWin 'pico_canonical_initial_fixture_v1/initial_integration_state.npz') '4f99f1bd0d9559ec23bd0a73afe5e49a711a90c3dcecafd1a121bae89b0dbde3'
Assert-Pin (Join-Path $baseWin 'pico_hold_independent_fixture_v1/initial_integration_state.npz') 'e9f4a6361be91a29711558ab8a6baea110caf2b86da10d643b22a26c1003b006'
$qualification = Get-Content -Raw -LiteralPath $qualificationPath | ConvertFrom-Json
if (($qualification.complete_offline_pico_pass -ne $true) -or ($qualification.both_quiet_windows_pass -ne $true)) {
    throw 'The bound full lifecycle and separate hold must already be qualified.'
}
foreach ($entry in $qualification.traces.PSObject.Properties) {
    Assert-Pin $entry.Value.path $entry.Value.sha256
}
foreach ($entry in $qualification.independent_reports.PSObject.Properties) {
    Assert-Pin $entry.Value.path $entry.Value.sha256
}
foreach ($entry in (Get-Content -Raw -LiteralPath (Join-Path $baseWin 'preserved_walk_demo_v1/mapping.json') | ConvertFrom-Json)) {
    Assert-Pin $entry.frozen $entry.sha256
}
New-Item -ItemType Directory -Path $outputWin | Out-Null

function Invoke-RecordedReplay([string]$Part, [string]$Producer, [string]$Fixture, [int]$Controls) {
    $arguments = @(
        '-d', 'Ubuntu-22.04', '--cd', $repoWsl,
        '--', 'bash', "$repoWsl/artifacts/teleop_six_hour_20260910/RUN_PASSING_WALK_WSL.sh",
        'env', 'OMP_NUM_THREADS=1', 'OPENBLAS_NUM_THREADS=1', 'MKL_NUM_THREADS=1',
        '/mnt/e/codex_sonic_runtime/mjbatch323_20260910/venv/bin/python',
        'artifacts/teleop_resume_20260911/audit_restored_native_segment.py',
        '--frozen-repo', "$baseWsl/preserved_walk_demo_v1/repo",
        '--oracle', 'gear_sonic/utils/g1_true23_feasibility_referee.py',
        '--bundle', 'artifacts/teleop_six_hour_20260910/mjbatch_native23_inputs_v1',
        '--fixture', "$baseWsl/$Fixture/initial_integration_state.npz",
        '--trace', "$baseWsl/$Producer/trace.npz", '--clip', 'pico',
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

Invoke-RecordedReplay 'lifecycle' 'pico_full_control_lm_v1' 'pico_canonical_initial_fixture_v1' 6530
Invoke-RecordedReplay 'separate_hold' 'pico_terminal_hold_v1' 'pico_hold_independent_fixture_v1' 250
$result = [ordered]@{
    kind = 'qualified_pico_recorded_command_native_physics_replay'
    qualification_sha256 = Read-Sha256 $qualificationPath
    lifecycle_physics_steps = 65300
    separate_hold_physics_steps = 2500
    every_recorded_sample_bitexact = $true
    new_controller_inference = $false
    live_teleoperation_qualified = $false
    completed_utc = [DateTime]::UtcNow.ToString('o')
}
$result | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $outputWin 'replay_result.json')
Write-Output 'PASS: all 67,800 recorded native physics steps match exactly.'
Write-Output "Evidence: $outputWin"
Write-Output 'Recorded-command replay only; live teleoperation remains unqualified.'
