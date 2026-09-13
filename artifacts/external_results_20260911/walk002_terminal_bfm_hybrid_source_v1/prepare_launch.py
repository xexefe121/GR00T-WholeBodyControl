"""Freeze selected single-run inputs and durable hidden launcher, no execution."""
import hashlib
import json
from pathlib import Path

SRC = Path(__file__).parent
NEW = SRC.parent
ROOT = Path('Z:/codex/GR00T-WholeBodyControl-sonic-transfer-23dof')
PROCESS = NEW/'walk002_terminal_bfm_hybrid_v1_process'
PROCESS.mkdir(exist_ok=True)

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def wsl(p):
    p=str(p).replace('\\','/')
    return '/mnt/'+p[0].lower()+p[2:]
def write(p,obj):p.write_text(json.dumps(obj,indent=2)+'\n')

inputs = {Path(p):digest for p,digest in json.loads((NEW/'walk002_full_control_lm_v1_process/launch_receipt.json').read_text())['verified_hashes'].items()}
extra = list((SRC/'repo').rglob('*.py')) + list((SRC/'repo').rglob('*.json'))
extra += [SRC/p for p in ('runner.py','hybrid_checks.py','test_hybrid.py','focused_final.xml','import_hashes.json','preflight_readonly.py')]
for folder,names in (
    ('walk002_full_control_lm_v1',('trace.npz','request.json','report.json')),
    ('walk002_full_control_lm_independent_physics_v1',('report.json',)),
    ('walk002_full_control_lm_independent_intent_v1',('report.json',)),
    ('walk002_preterminal1117_endpoint_independent_v1',('endpoint.npz','report.json','source_snapshot.py'))):
    for name in names:
        p=NEW/folder/name
        if p.exists():extra.append(p)
        elif name!='source_snapshot.py':raise FileNotFoundError(p)
extra += [p for p in (NEW/'walk002_preterminal1117_endpoint_independent_v1').glob('*.py')]
for p in extra:inputs[p]=sha(p)
for p,d in inputs.items():assert sha(p)==d,str(p)
manifest=SRC/'runtime_manifest.json'
write(manifest,dict(kind='one_selected_offline_walk002_saved_prefix_BFM_hybrid',hashes={wsl(p):d for p,d in inputs.items()},
    original_main_quiet_result='FAIL preserved',source_controls=667,prefix_controls=1117,original_terminal_controls=300,separate_hold_controls=250,
    frozen_import_root=wsl(SRC/'repo'),root_boundary_comparison_only=True))

params=dict(repo=SRC/'repo',bundle=ROOT/'artifacts/teleop_six_hour_20260910/mjbatch_native23_inputs_v1',
    reference=Path('E:/codex-artifacts/sonic23_teleop_six_hour_20260910/mjbatch_intent_floor_inputs_v1/walk002/reference.npz'),
    producer=NEW/'walk002_full_control_lm_v1',physical_audit=NEW/'walk002_full_control_lm_independent_physics_v1',
    intent_audit=NEW/'walk002_full_control_lm_independent_intent_v1/report.json',
    boundary_endpoint=NEW/'walk002_preterminal1117_endpoint_independent_v1/endpoint.npz',
    onnx=ROOT/'artifacts/teleop_six_hour_20260910/bfm_onnx_v2',
    dependencies=Path('E:/codex_sonic_runtime/bfm_seed_20260910/onnx_deps'),manifest=manifest,output=NEW/'walk002_terminal_bfm_hybrid_v1')
write(SRC/'arguments.json',dict(**{k:wsl(p) for k,p in params.items()},clip='walk002',switch_control=1117,requested_controls=1417,extension_controls=250))
launcher=NEW/'launch_walk002_terminal_bfm_hybrid_v1.ps1'
command=['wsl.exe','-d','Ubuntu-22.04','--cd',wsl(ROOT),'--','bash',wsl(ROOT/'artifacts/teleop_six_hour_20260910/RUN_PASSING_WALK_WSL.sh'),
    'env','PYTHONPATH='+wsl(SRC/'repo'),'OMP_NUM_THREADS=1','OPENBLAS_NUM_THREADS=1','MKL_NUM_THREADS=1',
    '/mnt/e/codex_sonic_runtime/mjbatch323_20260910/venv/bin/python',wsl(SRC/'runner.py')]
for k,v in params.items():command.extend(['--'+k.replace('_','-'),wsl(v)])
command += ['--clip','walk002','--switch-control','1117','--requested-controls','1417','--extension-controls','250']
stdout=NEW/'walk002_terminal_bfm_hybrid_v1.stdout.log';stderr=NEW/'walk002_terminal_bfm_hybrid_v1.stderr.log'
quote=lambda s:"'"+str(s).replace("'","''")+"'"
launcher.write_text("$ErrorActionPreference = 'Stop'\n& "+' '.join(quote(s) for s in command)+
    ' 1> '+quote(stdout)+' 2> '+quote(stderr)+"\nexit $LASTEXITCODE\n")
durable=PROCESS/'run_durable.ps1'
durable.write_text(r'''$ErrorActionPreference = 'Stop'
$base = 'E:\codex-artifacts\sonic23_teleop_resume_20260911'
$processDirectory = Join-Path $base 'walk002_terminal_bfm_hybrid_v1_process'
$launcher = Join-Path $base 'launch_walk002_terminal_bfm_hybrid_v1.ps1'
$started = [DateTime]::UtcNow.ToString('o')
$exitStatus = 1
$errorMessage = $null
function Read-TaskSha256([string]$Path) {
    $stream = [System.IO.File]::OpenRead($Path)
    $hasher = [System.Security.Cryptography.SHA256]::Create()
    try { return ([BitConverter]::ToString($hasher.ComputeHash($stream))).Replace('-','').ToLowerInvariant() }
    finally { $hasher.Dispose(); $stream.Dispose() }
}
try {
    $receipt = Get-Content -LiteralPath (Join-Path $processDirectory 'launch_receipt.json') -Raw | ConvertFrom-Json
    foreach ($entry in $receipt.verified_hashes.PSObject.Properties) {
        if ((Read-TaskSha256 $entry.Name) -ne $entry.Value) { throw ('Input hash mismatch: ' + $entry.Name) }
    }
    foreach ($name in @('walk002_terminal_bfm_hybrid_v1','walk002_terminal_bfm_hybrid_v1.stdout.log','walk002_terminal_bfm_hybrid_v1.stderr.log')) {
        if (Test-Path -LiteralPath (Join-Path $base $name)) { throw ('Output exists; refusing duplicate run: ' + $name) }
    }
    @{state='running';pid=$PID;started_utc=$started;launcher=$launcher;requested_controls=1417;separate_hold_controls=250} | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $processDirectory 'running.json')
    $shellPath = (Get-Process -Id $PID).Path
    & $shellPath -NoProfile -NonInteractive -File $launcher
    $exitStatus = $LASTEXITCODE
} catch {
    $errorMessage = $_.Exception.ToString()
    Write-Error $errorMessage -ErrorAction Continue
} finally {
    $comparisonPath = Join-Path $base 'walk002_terminal_bfm_hybrid_v1\comparison.json'
    $result = @{state='exited';pid=$PID;started_utc=$started;finished_utc=[DateTime]::UtcNow.ToString('o');exit_code=$exitStatus;error=$errorMessage;comparison_exists=(Test-Path -LiteralPath $comparisonPath);completed=$false;requested_controls=1417;separate_hold_requested_controls=250}
    if ($result.comparison_exists) {
        $report = Get-Content -LiteralPath $comparisonPath -Raw | ConvertFrom-Json
        $result.physical_completed = $report.lifecycle.completed -and $null -ne $report.extension -and $report.extension.completed
        $result.main_quiet_pass = $report.lifecycle.quiet_pass
        $result.hold_quiet_pass = $null -ne $report.extension -and $report.extension.quiet_pass
        $result.completed = $result.physical_completed -and $result.main_quiet_pass -and $result.hold_quiet_pass -and $report.inputs_unchanged
        if (-not $result.completed) { $exitStatus = 1; $result.exit_code = 1 }
        $result.completed_controls = $report.lifecycle.completed_controls
        $result.hold_completed_controls = $report.extension.completed_controls
        $result.failure = $report.lifecycle.failure
        $result.comparison_sha256 = Read-TaskSha256 $comparisonPath
    }
    $temp = Join-Path $processDirectory 'exit_status.tmp.json'
    $result | ConvertTo-Json -Depth 40 | Set-Content -LiteralPath $temp
    Move-Item -LiteralPath $temp -Destination (Join-Path $processDirectory 'exit_status.json')
}
exit $exitStatus
''')
for p in (manifest,SRC/'arguments.json',launcher,durable):inputs[p]=sha(p)
write(PROCESS/'launch_receipt.json',dict(kind='one_selected_walk002_terminal_BFM_hybrid',command=command,
    requested_controls=1417,prefix_controls=1117,original_BFM_terminal_controls=300,separate_hold_controls=250,
    source_tests_xml=sha(SRC/'focused_final.xml'),verified_hashes={str(p):d for p,d in inputs.items()},
    hidden_durable_required=True,execution_started=False,original_quiet_failure_preserved=True))
print(json.dumps(dict(manifest_sha256=sha(manifest),launcher_sha256=sha(launcher),durable_sha256=sha(durable),
    launch_receipt_sha256=sha(PROCESS/'launch_receipt.json'),bound_files=len(inputs))))
