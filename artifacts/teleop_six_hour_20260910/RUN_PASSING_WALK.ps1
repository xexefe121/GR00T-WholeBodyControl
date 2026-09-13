# Offline native23 physical replay plus tested BFM terminal standing.
# No robot, DDS, live Pico transport or online-MPC claim.
param(
    [ValidatePattern('^[A-Za-z0-9_-]+$')]
    [string]$RunName = ('walk003_quiet_replay_' + (Get-Date -Format 'yyyyMMdd_HHmmss'))
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
$repoWin = 'Z:\codex\GR00T-WholeBodyControl-sonic-transfer-23dof'
$repoWsl = '/mnt/z/codex/GR00T-WholeBodyControl-sonic-transfer-23dof'
$frozenRepoWsl = '/mnt/e/codex-artifacts/sonic23_teleop_resume_20260911/preserved_walk_demo_v1/repo'
$frozenMappingPath = 'E:\codex-artifacts\sonic23_teleop_resume_20260911\preserved_walk_demo_v1\mapping.json'
$archiveWin = 'E:\codex-artifacts\sonic23_teleop_six_hour_20260910'
$archiveWsl = '/mnt/e/codex-artifacts/sonic23_teleop_six_hour_20260910'
$evidenceWin = Join-Path $archiveWin 'bfm_online_intent_v2/walk003_terminal_bfm_yaw4_hybrid_v1'
$evidenceWsl = "$archiveWsl/bfm_online_intent_v2/walk003_terminal_bfm_yaw4_hybrid_v1"
$outputWin = Join-Path $archiveWin "bfm_online_intent_v2/$RunName"
$outputWsl = "$archiveWsl/bfm_online_intent_v2/$RunName"
$launcherLog = Join-Path $archiveWin "bfm_online_intent_v2/$RunName.launcher.log"
if ((Test-Path -LiteralPath $outputWin) -or (Test-Path -LiteralPath $launcherLog)) {
    throw 'Choose a new RunName. Existing evidence is never overwritten.'
}

# Imports use an isolated copy of the passing code while development continues.
# Original model and recording assets remain hash-pinned in their original paths.
$frozenMapping = @{}
foreach ($item in (Get-Content -Raw -LiteralPath $frozenMappingPath | ConvertFrom-Json)) {
    if ((Get-FileHash -LiteralPath $item.frozen -Algorithm SHA256).Hash -ne $item.sha256) {
        throw "Frozen demo source changed: $($item.frozen)"
    }
    $frozenMapping[[IO.Path]::GetFullPath($item.original)] = $item.frozen
}
$provenance = Get-Content -Raw -LiteralPath (Join-Path $evidenceWin 'provenance.json') | ConvertFrom-Json
foreach ($entry in $provenance.hashes.PSObject.Properties) {
    if ($entry.Name.StartsWith('/mnt/z/')) {
        $inputWin = 'Z:\' + $entry.Name.Substring(7).Replace('/', '\')
    } elseif ($entry.Name.StartsWith('/mnt/e/')) {
        $inputWin = 'E:\' + $entry.Name.Substring(7).Replace('/', '\')
    } else {
        throw "Unrecognized pinned input path: $($entry.Name)"
    }
    $originalInputWin = [IO.Path]::GetFullPath($inputWin)
    if ($frozenMapping.ContainsKey($originalInputWin)) { $inputWin = $frozenMapping[$originalInputWin] }
    if ((Get-FileHash -LiteralPath $inputWin -Algorithm SHA256).Hash -ne $entry.Value) {
        throw "Pinned input changed: $inputWin"
    }
}

& wsl.exe -d Ubuntu-22.04 --cd $repoWsl -- bash `
    "$repoWsl/artifacts/teleop_six_hour_20260910/RUN_PASSING_WALK_WSL.sh" env `
    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 `
    /mnt/e/codex_sonic_runtime/mjbatch323_20260910/venv/bin/python `
    "$evidenceWsl/runner_snapshot.py" `
    --repo $frozenRepoWsl `
    --bundle "$repoWsl/artifacts/teleop_six_hour_20260910/mjbatch_native23_inputs_v1" `
    --reference "$archiveWsl/mjbatch_intent_floor_inputs_v1/walk003/reference.npz" `
    --producer "$archiveWsl/mjbatch_full_v1/walk003_v4_native323_allmargin_full_v1" `
    --baseline "$archiveWsl/bfm_online_intent_v2/walk003_allmargin_wsl_independent_replay_v1" `
    --onnx "$repoWsl/artifacts/teleop_six_hour_20260910/bfm_onnx_v2" `
    --dependencies /mnt/e/codex_sonic_runtime/bfm_seed_20260910/onnx_deps `
    --output $outputWsl 1> $launcherLog
if ($LASTEXITCODE -ne 0) { throw "Physical replay failed. See $launcherLog" }

Push-Location -LiteralPath $repoWin
try {
    & 'C:\Users\camer\AppData\Local\Programs\Python\Python310\python.exe' `
        -m artifacts.teleop_six_hour_20260910.qualify_recorded_candidate `
        $outputWin `
        --motion-override "$archiveWin/mjbatch_intent_floor_inputs_v1/walk003/reference.npz" `
        --output "$outputWin/recorded_source_audit_v2.json" --require-pass `
        1> "$outputWin/source_audit.stdout.txt"
    if ($LASTEXITCODE -ne 0) { throw "Recorded-source acceptance failed: $outputWin" }
} finally {
    Pop-Location
}

$comparison = Get-Content -Raw -LiteralPath (Join-Path $outputWin 'comparison.json') | ConvertFrom-Json
if (($comparison.lifecycle.quiet_standing_diagnostic.quiet_standing_diagnostic_pass -ne $true) -or
    ($comparison.extension.quiet_standing_diagnostic.quiet_standing_diagnostic_pass -ne $true)) {
    throw "Quiet-standing diagnostic failed: $outputWin"
}
Write-Output "PASS: complete recorded-source replay, lifecycle and separate 5-second quiet hold."
Write-Output "Evidence: $outputWin"
Write-Output 'Offline replay only. Live teleoperation and hardware remain unqualified.'
