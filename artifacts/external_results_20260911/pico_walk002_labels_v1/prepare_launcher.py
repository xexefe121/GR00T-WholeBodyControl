"""Create a durable hidden collection command bound to the final frozen inputs."""
import hashlib
import json
from pathlib import Path

BASE = Path(__file__).parent
PROCESS = BASE / 'process'
PROCESS.mkdir(exist_ok=True)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def quote(value):
    return "'" + str(value).replace("'", "''") + "'"


manifest = BASE / 'collector_frozen_inputs.json'
frozen = json.loads(manifest.read_text())
assert frozen['frozen'] and not (BASE / 'collection').exists()
arguments = ['-d','Ubuntu-22.04','--cd','/mnt/z/codex/GR00T-WholeBodyControl-sonic-transfer-23dof','--',
    'bash','/mnt/z/codex/GR00T-WholeBodyControl-sonic-transfer-23dof/artifacts/teleop_six_hour_20260910/RUN_PASSING_WALK_WSL.sh',
    'env','OMP_NUM_THREADS=1','OPENBLAS_NUM_THREADS=1','MKL_NUM_THREADS=1',
    'PYTHONPATH=/mnt/e/codex-artifacts/sonic23_teleop_resume_20260911/pico_walk002_labels_v1/source_snapshot_v1',
    '/mnt/e/codex_sonic_runtime/mjbatch323_20260910/venv/bin/python',
    '/mnt/e/codex-artifacts/sonic23_teleop_resume_20260911/pico_walk002_labels_v1/source_snapshot_v1/collect_qualified_rows.py',
    '--manifest','/mnt/e/codex-artifacts/sonic23_teleop_resume_20260911/pico_walk002_labels_v1/collector_frozen_inputs.json']
launcher = PROCESS / 'run_collection.ps1'
launcher.write_text("""$ErrorActionPreference = 'Stop'
function Get-TaskHash([string] $Path) {
    $algorithm = [System.Security.Cryptography.SHA256]::Create()
    try { return ([System.BitConverter]::ToString($algorithm.ComputeHash([System.IO.File]::ReadAllBytes($Path)))).Replace('-', '').ToLowerInvariant() }
    finally { $algorithm.Dispose() }
}
""" + '$manifestPath = ' + quote(manifest) + '\n' +
    "if ((Get-TaskHash $manifestPath) -ne " + quote(sha(manifest)) + ") { throw 'Frozen manifest changed.' }\n" +
    "$frozen = Get-Content -Raw -LiteralPath $manifestPath | ConvertFrom-Json\n" +
    "foreach ($entry in $frozen.input_hashes.PSObject.Properties) { if ((Get-TaskHash $entry.Name) -ne $entry.Value) { throw ('Input changed: ' + $entry.Name) } }\n" +
    '$arguments = @(\n' + ',\n'.join('    '+quote(a) for a in arguments) + '\n)\n' +
    '& "$env:SystemRoot\\System32\\wsl.exe" @arguments\n' +
    '$code = $LASTEXITCODE\n' +
    "if ($code -ne 0) { exit $code }\n" +
    '$reportPath = ' + quote(BASE / 'collection/report.json') + '\n' +
    "$result = Get-Content -Raw -LiteralPath $reportPath | ConvertFrom-Json\n" +
    "if (-not $result.complete -or $result.completed_rows -ne 6847 -or $result.actor_inference_calls -ne 6847 -or $result.physics_steps -ne 0 -or $result.fitting_launched) { throw 'Incomplete or invalid collection result.' }\n" +
    "exit 0\n")
durable = PROCESS / 'run_collection_durable.ps1'
durable.write_text("""$ErrorActionPreference = 'Stop'
function Get-TaskHash([string] $Path) {
    $algorithm = [System.Security.Cryptography.SHA256]::Create()
    try { return ([System.BitConverter]::ToString($algorithm.ComputeHash([System.IO.File]::ReadAllBytes($Path)))).Replace('-', '').ToLowerInvariant() }
    finally { $algorithm.Dispose() }
}
""" + '$processFolder = ' + quote(PROCESS) + '\n' +
    "$receiptPath = Join-Path $processFolder 'launch_receipt.json'\n" +
    "$receipt = Get-Content -Raw -LiteralPath $receiptPath | ConvertFrom-Json\n" +
    "foreach ($entry in $receipt.input_hashes.PSObject.Properties) { if ((Get-TaskHash $entry.Name) -ne $entry.Value) { throw ('Launch input changed: ' + $entry.Name) } }\n" +
    "$lockPath = Join-Path $processFolder 'started.lock'\n" +
    "$lock = [System.IO.File]::Open($lockPath, [System.IO.FileMode]::CreateNew, [System.IO.FileAccess]::Write, [System.IO.FileShare]::None)\n$lock.Close()\n" +
    "$start = [ordered]@{ utc = [DateTime]::UtcNow.ToString('o'); wrapper_pid = $PID; receipt_sha256 = Get-TaskHash $receiptPath; intended_rows = 6847; inference_only = $true }\n" +
    "$start | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $processFolder 'start.json') -Encoding UTF8\n" +
    "$exitCode = 1\n$errorText = $null\ntry {\n" +
    "    $child = Start-Process -FilePath \"$env:SystemRoot\\System32\\WindowsPowerShell\\v1.0\\powershell.exe\" -ArgumentList @('-NoProfile', '-ExecutionPolicy', 'Bypass', '-File', (Join-Path $processFolder 'run_collection.ps1')) -WindowStyle Hidden -PassThru -RedirectStandardOutput (Join-Path $processFolder 'stdout.log') -RedirectStandardError (Join-Path $processFolder 'stderr.log')\n" +
    "    [ordered]@{ child_pid = $child.Id; wrapper_pid = $PID; utc = [DateTime]::UtcNow.ToString('o') } | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $processFolder 'child.json') -Encoding UTF8\n" +
    "    $child.WaitForExit()\n    $child.Refresh()\n    $exitCode = $child.ExitCode\n" +
    "    $postHashes = [ordered]@{}\n    foreach ($entry in $receipt.input_hashes.PSObject.Properties) { $actual = Get-TaskHash $entry.Name; $postHashes[$entry.Name] = $actual; if ($actual -ne $entry.Value) { throw ('Postrun input changed: ' + $entry.Name) } }\n" +
    "    $postHashes | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath (Join-Path $processFolder 'postrun_hashes.json') -Encoding UTF8\n" +
    "} catch { $errorText = $_.Exception.Message; $exitCode = 1 } finally {\n" +
    "    [ordered]@{ utc = [DateTime]::UtcNow.ToString('o'); exit_code = $exitCode; error = $errorText; intended_rows = 6847 } | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $processFolder 'exit.json') -Encoding UTF8\n" +
    "}\nexit $exitCode\n")
pins = dict(frozen['input_hashes'])
for p in (manifest, BASE/'preflight_frozen.json', BASE/'focused_final.xml', Path(__file__), launcher, durable,
          Path('Z:/codex/GR00T-WholeBodyControl-sonic-transfer-23dof/artifacts/teleop_six_hour_20260910/RUN_PASSING_WALK_WSL.sh')):
    pins[str(p).replace('\\','/')] = sha(p)
receipt = dict(kind='one_6847_row_inference_only_collection_launch', intended_rows=6847,
    root_selected_extraction=True, model_fitting_authorized=False, physics_authorized=False,
    exact_wsl_arguments=arguments, manifest_sha256=sha(manifest), input_hashes=pins)
path = PROCESS/'launch_receipt.json'
path.write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps(dict(launch_receipt_sha256=sha(path), hashes=len(pins))))
