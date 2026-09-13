"""Create one immutable hidden-launch package after actual binding validation.

Creating files launches nothing. --final-review binds a later actual reviewer
receipt; the durable runner refuses execution before this separate clearance.
"""
import argparse
import json
from pathlib import Path
import sys

BASE=Path(__file__).resolve().parent;SOURCE=BASE/'source_draft_v1'
sys.path.insert(0,str(SOURCE))
from evaluation_gate import read,sha,field,has_hash,require_model_ready,require_ready

WINPY='C:/Users/camer/AppData/Local/Programs/Python/Python310/python.exe'
BOOTSTRAP='Z:/codex/GR00T-WholeBodyControl-sonic-transfer-23dof/artifacts/teleop_six_hour_20260910/RUN_PASSING_WALK_WSL.sh'
HASH="""$ErrorActionPreference = 'Stop'
function Get-TaskHash([string] $Path) {
    $algorithm = [System.Security.Cryptography.SHA256]::Create()
    $stream = [System.IO.File]::Open($Path,[System.IO.FileMode]::Open,[System.IO.FileAccess]::Read,([System.IO.FileShare]::ReadWrite -bor [System.IO.FileShare]::Delete))
    try { return ([System.BitConverter]::ToString($algorithm.ComputeHash($stream))).Replace('-', '').ToLowerInvariant() }
    finally { $stream.Dispose(); $algorithm.Dispose() }
}
function Write-NewJson([string] $Path, $Value) {
    $text = $Value | ConvertTo-Json -Depth 12
    $stream = [System.IO.File]::Open($Path,[System.IO.FileMode]::CreateNew,[System.IO.FileAccess]::Write,[System.IO.FileShare]::Read)
    try { $bytes = [System.Text.Encoding]::UTF8.GetBytes($text); $stream.Write($bytes,0,$bytes.Length) } finally { $stream.Dispose() }
}
"""

def quote(value):return "'"+str(value).replace("'","''")+"'"
def write_new(path,value):
    with Path(path).open('x',encoding='utf-8') as f:f.write(value)
def write_json(path,value):write_new(path,json.dumps(value,indent=2,allow_nan=False)+'\n')

def wsl_arguments(base,mode):
    linux='/mnt/e/codex-artifacts/sonic23_teleop_resume_20260911/'+base.name
    script='head_activation_witness.py' if mode=='witness' else 'evaluate_direct_target_student.py'
    return ['-d','Ubuntu-22.04','--cd','/','--','bash','/mnt/z'+BOOTSTRAP[2:],'env',
        'OMP_NUM_THREADS=1','OPENBLAS_NUM_THREADS=1','MKL_NUM_THREADS=1',
        'PYTHONDONTWRITEBYTECODE=1','PYTHONPATH='+linux+'/source_draft_v1','/mnt/e/codex_sonic_runtime/mjbatch323_20260910/venv/bin/python','-B',
        linux+'/source_draft_v1/'+script]

def run_text(base,folder,mode,binding_sha):
    args=wsl_arguments(base,mode)
    return HASH+'$folder = '+quote(folder)+'\n'+\
        '$bindingPath = '+quote(base/(mode+'_binding.json'))+'\n'+\
        'if ((Get-TaskHash $bindingPath) -ne '+quote(binding_sha)+") { throw 'Final binding changed.' }\n"+\
        "$binding = Get-Content -Raw -LiteralPath $bindingPath | ConvertFrom-Json\n"+\
        "foreach ($entry in $binding.input_files) { if ((Get-TaskHash $entry.path) -ne $entry.sha256) { throw ('Bound input changed: ' + $entry.path) } }\n"+\
        '$arguments = @(\n'+',\n'.join('    '+quote(a) for a in args)+'\n)\n'+\
        "$rawExit = $null\n$rawError = $null\ntry {\n"+\
        '    & "$env:SystemRoot\\System32\\wsl.exe" @arguments\n    $rawExit = $LASTEXITCODE\n'+\
        "} catch { $rawError = $_.Exception.Message } finally {\n"+\
        "    Write-NewJson (Join-Path $folder 'raw_exit.json') ([ordered]@{utc=[DateTime]::UtcNow.ToString('o');raw_python_exit_code=$rawExit;raw_error=$rawError;known=($null -ne $rawExit)})\n}\n"+\
        '& '+quote(WINPY)+' '+quote(base/'diagnostic_verdict.py')+' --base '+quote(base)+' --mode '+quote(mode)+\
        " --raw-record (Join-Path $folder 'raw_exit.json') --output (Join-Path $folder 'diagnostic_verdict.json')\n"+\
        "$verdictExit = $LASTEXITCODE\nif ($null -eq $verdictExit) { throw 'Diagnostic exit unavailable.' }\nexit $verdictExit\n"

def durable_text(folder):
    return HASH+'$folder = '+quote(folder)+'\n'+r"""$receiptPath = Join-Path $folder 'launch_receipt.json'
$receipt = Get-Content -Raw -LiteralPath $receiptPath | ConvertFrom-Json
$clearancePath = Join-Path $folder 'launch_clearance.json'
$clearance = Get-Content -Raw -LiteralPath $clearancePath | ConvertFrom-Json
if ($clearance.launch_receipt_sha256 -ne (Get-TaskHash $receiptPath) -or -not $clearance.selected_single_run) { throw 'Actual final launch clearance missing or changed.' }
if ((Get-TaskHash $clearance.review.path) -ne $clearance.review.sha256) { throw 'Final reviewer receipt changed.' }
$review = Get-Content -Raw -LiteralPath $clearance.review.path | ConvertFrom-Json
$pass = $review
foreach ($key in $clearance.review.pass_field.Split('.')) { $pass = $pass.$key }
if ($pass -ne $true) { throw 'Final reviewer gate not passed.' }
if ($review.launch_receipt_subject.sha256 -ne $clearance.launch_receipt_sha256 -or $review.launch_receipt_subject.path -ne $receiptPath.Replace([char]92,[char]47) -or $review.binding_subject.sha256 -ne $receipt.binding_sha256 -or $review.binding_subject.path -ne $receipt.binding_path) { throw 'Final review does not name this exact binding and launch.' }
foreach ($entry in $receipt.input_hashes.PSObject.Properties) { if ((Get-TaskHash $entry.Name) -ne $entry.Value) { throw ('Launch input changed: ' + $entry.Name) } }
$lock = [System.IO.File]::Open((Join-Path $folder 'started.lock'),[System.IO.FileMode]::CreateNew,[System.IO.FileAccess]::Write,[System.IO.FileShare]::None)
$lock.Close()
Write-NewJson (Join-Path $folder 'start.json') ([ordered]@{utc=[DateTime]::UtcNow.ToString('o');wrapper_pid=$PID;receipt_sha256=(Get-TaskHash $receiptPath);clearance_sha256=(Get-TaskHash $clearancePath);review_sha256=(Get-TaskHash $clearance.review.path)})
$exitCode = 1
$childExit = $null
$errorText = $null
try {
    $taskChild = Start-Process -FilePath "$env:SystemRoot\System32\WindowsPowerShell\v1.0\powershell.exe" -ArgumentList @('-NoProfile','-ExecutionPolicy','Bypass','-File',(Join-Path $folder 'run.ps1')) -WindowStyle Hidden -PassThru -RedirectStandardOutput (Join-Path $folder 'stdout.log') -RedirectStandardError (Join-Path $folder 'stderr.log')
    $nativeHandle = $taskChild.Handle
    Write-NewJson (Join-Path $folder 'child.json') ([ordered]@{child_pid=$taskChild.Id;wrapper_pid=$PID;handle_acquired=($nativeHandle -ne [IntPtr]::Zero)})
    $taskChild.WaitForExit()
    $childExit = $taskChild.ExitCode
    if ($null -eq $childExit) { throw 'Child exit status unavailable.' }
    $exitCode = $childExit
} catch { $errorText=$_.Exception.Message; $exitCode=1 } finally {
    $post = [ordered]@{}
    $allExact = $true
    foreach ($entry in $receipt.input_hashes.PSObject.Properties) {
        try { $actual = Get-TaskHash $entry.Name } catch { $actual = $null }
        $post[$entry.Name] = $actual
        if ($actual -ne $entry.Value) { $allExact=$false }
    }
    Write-NewJson (Join-Path $folder 'postrun_hashes.json') $post
    if (-not $allExact) { $exitCode=1; if ($null -eq $errorText) { $errorText='One or more postrun inputs changed.' } }
    if ((Get-TaskHash $clearancePath) -ne (Get-Content -Raw -LiteralPath (Join-Path $folder 'start.json') | ConvertFrom-Json).clearance_sha256) { $exitCode=1; $errorText='Launch clearance changed.' }
    Write-NewJson (Join-Path $folder 'exit.json') ([ordered]@{utc=[DateTime]::UtcNow.ToString('o');exit_code=$exitCode;raw_child_exit_code=$childExit;error=$errorText;all_postrun_hashes_exact=$allExact;automatic_retry=$false})
}
exit $exitCode
"""

def main():
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=['witness','evaluation'],required=True)
    p.add_argument('--final-review',type=Path);p.add_argument('--pass-field',default='passed');a=p.parse_args()
    if a.mode=='witness':require_model_ready(BASE,'witness')
    else:require_ready(BASE)
    binding_path=BASE/(a.mode+'_binding.json');binding=read(binding_path);folder=BASE/(a.mode+'_process')
    if a.final_review:
        receipt=folder/'launch_receipt.json';review=read(a.final_review)
        assert field(review,a.pass_field) is True
        require_review_subjects(review,binding_path,receipt)
        assert not (folder/'started.lock').exists()
        write_json(folder/'launch_clearance.json',dict(selected_single_run=True,launch_receipt_sha256=sha(receipt),
            review=dict(path=a.final_review.resolve().as_posix(),sha256=sha(a.final_review),pass_field=a.pass_field),
            hardware_authorized=False,automatic_retry=False))
        print(json.dumps(dict(clearance_sha256=sha(folder/'launch_clearance.json'),launched=False)));return
    folder.mkdir(exist_ok=False)
    exit_test=BASE.parent/'broader_labels_independent_v1/exit_capture_test.json'
    check=read(exit_test);assert check['actual_exit']==check['expected_exit']==7 and check['handle_acquired_before_wait'] is True
    launcher=folder/'run.ps1';durable=folder/'run_durable.ps1'
    write_new(launcher,run_text(BASE,folder,a.mode,sha(binding_path)));write_new(durable,durable_text(folder))
    pins={entry['path']:entry['sha256'] for entry in binding['input_files']}
    for path in (binding_path,Path(__file__),BASE/'diagnostic_verdict.py',launcher,durable,exit_test,Path(BOOTSTRAP),Path(WINPY)):
        pins[path.resolve().as_posix()]=sha(path)
    write_json(folder/'launch_receipt.json',dict(kind='one_selected_direct_target_'+a.mode,
        binding_path=binding_path.as_posix(),binding_sha256=sha(binding_path),exact_wsl_arguments=wsl_arguments(BASE,a.mode),input_hashes=pins,
        requested_main_controls=0 if a.mode=='witness' else 1569,conditional_hold_controls=0 if a.mode=='witness' else 250,
        expected_separate_head_calls=1 if a.mode=='witness' else 0,preparation_only=True,
        raw_python_exit_preserved=True,failed_or_incomplete_diagnostic_exit=2,hardware_authorized=False,automatic_retry=False))
    print(json.dumps(dict(launch_receipt_sha256=sha(folder/'launch_receipt.json'),pins=len(pins),launched=False)))

def require_review_subjects(review,binding,receipt):
    for name,path in [('binding_subject',binding),('launch_receipt_subject',receipt)]:
        entry=review[name]
        assert Path(entry['path']).resolve()==Path(path).resolve() and entry['sha256']==sha(path),name


if __name__=='__main__':main()
