$ErrorActionPreference = 'Continue'
$runRoot = 'E:\codex-artifacts\sonic23_teleop_resume_20260911\fresh_expert_labels_resume_v1'
$qualificationPath = 'E:\codex-artifacts\sonic23_teleop_resume_20260911\expert_resumed_root_qualification_v1\qualification.json'
$statePath = Join-Path $runRoot 'collection_process_status.json'
$status = [ordered]@{ kind='authorized_single_expert_label_collection'; state='COLLECTING'; powershell_pid=$PID; started_utc=[DateTime]::UtcNow.ToString('o'); physics_steps=0; fitting_launched=$false }
$status | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $statePath -Encoding UTF8
$runExit = -1
$runError = $null
function Read-Sha256([string]$path) {
    $digest = [Security.Cryptography.SHA256]::Create()
    try { return ([BitConverter]::ToString($digest.ComputeHash([IO.File]::ReadAllBytes($path)))).Replace('-', '').ToLowerInvariant() } finally { $digest.Dispose() }
}
try {
    if ((Read-Sha256 (Join-Path $runRoot 'collector_frozen_inputs.json')) -ne '0dd8902c46b84fc0cfbd12bbaec8e5b1a2875c5470171c29f6b4c954819c6ad3') { throw 'Reviewed collector receipt changed.' }
    if ((Read-Sha256 $qualificationPath) -ne '476408e209a598adafba06a1c1d7ebda715543005ade45ae4e6069c5bb3ee86f') { throw 'Root qualification receipt changed.' }
    & wsl.exe -d Ubuntu-22.04 --cd /mnt/z/codex/GR00T-WholeBodyControl-sonic-transfer-23dof -- bash /mnt/z/codex/GR00T-WholeBodyControl-sonic-transfer-23dof/artifacts/teleop_six_hour_20260910/RUN_PASSING_WALK_WSL.sh env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 /mnt/e/codex_sonic_runtime/mjbatch323_20260910/venv/bin/python -u /mnt/e/codex-artifacts/sonic23_teleop_resume_20260911/fresh_expert_labels_resume_v1/source_snapshot_v2/collect_actual_branch_labels.py --qualification /mnt/e/codex-artifacts/sonic23_teleop_resume_20260911/expert_resumed_root_qualification_v1/qualification.json 1> (Join-Path $runRoot 'collection_stdout.log') 2> (Join-Path $runRoot 'collection_stderr.log')
    $runExit = $LASTEXITCODE
    if ($runExit -ne 0) { throw "Collection exited $runExit; no retry or audit." }
    $status.state = 'AUDITING_SAVED_LABELS'
    $status['collection_exit_code'] = 0
    $status | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $statePath -Encoding UTF8
    & wsl.exe -d Ubuntu-22.04 --cd /mnt/z/codex/GR00T-WholeBodyControl-sonic-transfer-23dof -- bash /mnt/z/codex/GR00T-WholeBodyControl-sonic-transfer-23dof/artifacts/teleop_six_hour_20260910/RUN_PASSING_WALK_WSL.sh env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 /mnt/e/codex_sonic_runtime/mjbatch323_20260910/venv/bin/python -u /mnt/e/codex-artifacts/sonic23_teleop_resume_20260911/fresh_expert_labels_resume_v1/source_snapshot_v2/audit_label_compatibility.py 1> (Join-Path $runRoot 'compatibility_stdout.log') 2> (Join-Path $runRoot 'compatibility_stderr.log')
    $runExit = $LASTEXITCODE
} catch {
    $runError = $_.Exception.Message
} finally {
    $status.state = 'EXITED'
    $status['ended_utc'] = [DateTime]::UtcNow.ToString('o')
    $status['process_exit_code'] = $runExit
    $status['launcher_error'] = $runError
    $status | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $statePath -Encoding UTF8
}
exit $runExit
