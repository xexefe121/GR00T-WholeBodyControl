"""Freeze concrete commands without launching; final clearance requires root selection."""
import argparse,hashlib,json
from pathlib import Path

BASE=Path(__file__).resolve().parent
WINPY='C:/Users/camer/AppData/Local/Programs/Python/Python310/python.exe'
BOOT='Z:/codex/GR00T-WholeBodyControl-sonic-transfer-23dof/artifacts/teleop_six_hour_20260910/RUN_PASSING_WALK_WSL.sh'
PY='/mnt/e/codex_sonic_runtime/mjbatch323_20260910/venv/bin/python'

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def quote(value):return "'"+str(value).replace("'","''")+"'"
def write_new(path,text):
    with Path(path).open('x',encoding='utf-8') as f:f.write(text)
def write_json(path,value):write_new(path,json.dumps(value,indent=2,allow_nan=False)+'\n')
def linux(path):
    text=Path(path).as_posix();assert text[1]==':'
    return '/mnt/'+text[0].lower()+text[2:]

PS_HELPERS=r"""$ErrorActionPreference = 'Stop'
function Get-TaskHash([string] $Path) {
    $algorithm = [System.Security.Cryptography.SHA256]::Create()
    try { return ([System.BitConverter]::ToString($algorithm.ComputeHash([System.IO.File]::ReadAllBytes($Path)))).Replace('-', '').ToLowerInvariant() }
    finally { $algorithm.Dispose() }
}
function Write-NewJson([string] $Path, $Value) {
    $bytes = [System.Text.Encoding]::UTF8.GetBytes(($Value | ConvertTo-Json -Depth 16))
    $stream = [System.IO.File]::Open($Path,[System.IO.FileMode]::CreateNew,[System.IO.FileAccess]::Write,[System.IO.FileShare]::Read)
    try { $stream.Write($bytes,0,$bytes.Length) } finally { $stream.Dispose() }
}
"""

def arguments(stage):
    return ['-d','Ubuntu-22.04','--cd','/','--','bash',linux(BOOT),'env',
        'OMP_NUM_THREADS=1','OPENBLAS_NUM_THREADS=1','MKL_NUM_THREADS=1',
        'PYTHONPATH='+linux(BASE/'source_draft_v1'),PY,linux(BASE/'source_draft_v1/runner.py'),
        '--request',linux(BASE/(stage+'_request.json')),'--clearance',linux(BASE/(stage+'_process/launch_clearance.json'))]

def run_text(stage):
    folder=BASE/(stage+'_process')
    return PS_HELPERS+'$folder = '+quote(folder)+'\n$arguments = @(\n'+',\n'.join(quote(x) for x in arguments(stage))+r"""
)
$rawExit = $null
$rawError = $null
try {
    & "$env:SystemRoot\System32\wsl.exe" @arguments
    $rawExit = $LASTEXITCODE
} catch { $rawError = $_.Exception.Message } finally {
    Write-NewJson (Join-Path $folder 'raw_exit.json') ([ordered]@{utc=[DateTime]::UtcNow.ToString('o');raw_python_exit_code=$rawExit;raw_error=$rawError;known=($null -ne $rawExit)})
}
if ($null -eq $rawExit) { exit 1 }
exit $rawExit
"""

def durable_text(stage):
    folder=BASE/(stage+'_process')
    text=PS_HELPERS+'$folder = '+quote(folder)+'\n'+r"""$receiptPath = Join-Path $folder 'launch_receipt.json'
$receipt = Get-Content -Raw -LiteralPath $receiptPath | ConvertFrom-Json
$clearancePath = Join-Path $folder 'launch_clearance.json'
$clearance = Get-Content -Raw -LiteralPath $clearancePath | ConvertFrom-Json
if (-not $clearance.root_selected_single_run -or $clearance.launch_receipt_sha256 -ne (Get-TaskHash $receiptPath) -or $clearance.request_sha256 -ne $receipt.request_sha256) { throw 'Concrete root-selected launch clearance required.' }
if ((Get-TaskHash $clearance.review.path) -ne $clearance.review.sha256) { throw 'Final review changed.' }
$review = Get-Content -Raw -LiteralPath $clearance.review.path | ConvertFrom-Json
if ($review.($clearance.review.pass_field) -ne $true -or $review.request_subject.sha256 -ne $receipt.request_sha256 -or $review.request_subject.path -ne $receipt.request_path -or $review.launch_receipt_subject.sha256 -ne (Get-TaskHash $receiptPath) -or $review.launch_receipt_subject.path -ne $receiptPath.Replace('\','/')) { throw 'Final review does not name this exact request and launcher.' }
$pre = [ordered]@{}
foreach ($entry in $receipt.input_hashes.PSObject.Properties) {
    $actual = Get-TaskHash $entry.Name
    if ($actual -ne $entry.Value) { throw ('Input changed before launch: '+$entry.Name) }
    $pre[$entry.Name] = [ordered]@{expected=$entry.Value;actual=$actual;matched=$true}
}
$lock = [System.IO.File]::Open((Join-Path $folder 'started.lock'),[System.IO.FileMode]::CreateNew,[System.IO.FileAccess]::Write,[System.IO.FileShare]::None)
$lock.Close()
Write-NewJson (Join-Path $folder 'prerun_hashes.json') ([ordered]@{all_exact=$true;files=$pre})
Write-NewJson (Join-Path $folder 'start.json') ([ordered]@{utc=[DateTime]::UtcNow.ToString('o');wrapper_pid=$PID;receipt_sha256=(Get-TaskHash $receiptPath);clearance_sha256=(Get-TaskHash $clearancePath);review_sha256=(Get-TaskHash $clearance.review.path)})
$childExit = $null
$diagnosticExit = $null
$exitCode = 1
$errorText = $null
try {
    $taskChild = Start-Process -FilePath "$env:SystemRoot\System32\WindowsPowerShell\v1.0\powershell.exe" -ArgumentList @('-NoProfile','-ExecutionPolicy','Bypass','-File',(Join-Path $folder 'run.ps1')) -WindowStyle Hidden -PassThru -RedirectStandardOutput (Join-Path $folder 'stdout.log') -RedirectStandardError (Join-Path $folder 'stderr.log')
    $nativeHandle = $taskChild.Handle
    Write-NewJson (Join-Path $folder 'child.json') ([ordered]@{child_pid=$taskChild.Id;wrapper_pid=$PID;handle_acquired=($nativeHandle -ne [IntPtr]::Zero)})
    $taskChild.WaitForExit()
    $childExit = $taskChild.ExitCode
    if ($null -eq $childExit) { throw 'Child exit status unavailable; no automatic retry.' }
__VERDICT__
    $diagnosticExit = $LASTEXITCODE
    if ($null -eq $diagnosticExit) { throw 'Diagnostic exit unavailable.' }
    $exitCode = $diagnosticExit
    if ($childExit -ne 0 -and $exitCode -eq 0) { throw 'A nonzero native runner exit cannot pass.' }
} catch { $errorText=$_.Exception.Message; $exitCode=1 } finally {
    $post = [ordered]@{}
    $allExact = $true
    foreach ($entry in $receipt.input_hashes.PSObject.Properties) {
        try { $actual = Get-TaskHash $entry.Name } catch { $actual = $null }
        $post[$entry.Name] = [ordered]@{expected=$entry.Value;actual=$actual;matched=($actual -eq $entry.Value)}
        if ($actual -ne $entry.Value) { $allExact=$false }
    }
    Write-NewJson (Join-Path $folder 'postrun_hashes.json') ([ordered]@{all_exact=$allExact;files=$post})
    if (-not $allExact) { $exitCode=1; if ($null -eq $errorText) { $errorText='Postrun input changed.' } }
    if ((Get-TaskHash $clearancePath) -ne (Get-Content -Raw -LiteralPath (Join-Path $folder 'start.json') | ConvertFrom-Json).clearance_sha256 -or (Get-TaskHash $clearance.review.path) -ne $clearance.review.sha256) { $exitCode=1; $errorText='Final clearance or review changed.' }
    Write-NewJson (Join-Path $folder 'exit.json') ([ordered]@{utc=[DateTime]::UtcNow.ToString('o');exit_code=$exitCode;raw_python_exit_code=$childExit;diagnostic_exit_code=$diagnosticExit;known=($null -ne $childExit);error=$errorText;all_postrun_hashes_exact=$allExact;automatic_retry=$false})
}
exit $exitCode
"""
    verdict='    & '+quote(WINPY)+' '+quote(BASE/'stage_verdict.py')+' --base '+quote(BASE)+' --stage '+quote(stage)
    return text.replace('__VERDICT__',verdict)

def main():
    p=argparse.ArgumentParser();p.add_argument('--stage',choices=['witness','replay'],required=True)
    p.add_argument('--final-review',type=Path);p.add_argument('--pass-field',default='passed')
    p.add_argument('--root-selected',action='store_true');a=p.parse_args()
    request_path=BASE/(a.stage+'_request.json');folder=BASE/(a.stage+'_process')
    if a.final_review:
        assert a.root_selected,'Root must explicitly select this one stage before clearance.'
        review=read(a.final_review);receipt_path=folder/'launch_receipt.json'
        assert review[a.pass_field] is True
        for key,path in (('request_subject',request_path),('launch_receipt_subject',receipt_path)):
            assert Path(review[key]['path']).resolve()==path.resolve() and review[key]['sha256']==sha(path)
        assert not (folder/'started.lock').exists() and not (BASE/a.stage).exists()
        write_json(folder/'launch_clearance.json',dict(root_selected_single_run=True,
            request_sha256=sha(request_path),launch_receipt_sha256=sha(receipt_path),
            review=dict(path=a.final_review.resolve().as_posix(),sha256=sha(a.final_review),pass_field=a.pass_field),
            automatic_retry=False,hardware_authorized=False))
        print(json.dumps(dict(clearance_sha256=sha(folder/'launch_clearance.json'),launched=False)));return
    assert not a.root_selected,'Request preparation is separate from root selection and final clearance.'
    proposal=read(BASE/'proposal.json');pins={}
    for entry in proposal['input_files']:
        assert sha(entry['path'])==entry['sha256'],entry['path'];pins[entry['path']]=entry['sha256']
    preparation=read(BASE/'source_preparation.json')
    for path,digest in preparation['input_hashes'].items():assert sha(path)==digest; pins[path]=digest
    for path in (BASE/'proposal.json',BASE/'source_preparation.json',Path(__file__),BASE/'stage_verdict.py',BASE/'verify_completed_stage.py',Path(WINPY)):
        pins[path.resolve().as_posix()]=sha(path)
    if a.stage=='replay':
        owner=read(BASE/'witness_completion_verification.json');assert owner['passed'] is True
        for path,digest in owner['output_hashes'].items():assert sha(path)==digest; pins[path]=digest
        for path in (BASE/'witness_completion_verification.json',BASE/'witness_request.json'):
            pins[path.resolve().as_posix()]=sha(path)
    request=dict(stage=a.stage,source_preparation_only=False,execution_requires_separate_root_clearance=True,
        native_bundle=proposal['native_bundle'],canonical_fixture=proposal['canonical_fixture'],traces=proposal['traces'],
        native_step_budget=0 if a.stage=='witness' else 21348,serialization_budget=2 if a.stage=='witness' else 8,
        model_inference_calls=0,optimizer_updates=0,plant_foundation_connected=False,
        proposal_sha256=sha(BASE/'proposal.json'),source_preparation_sha256=sha(BASE/'source_preparation.json'),
        input_files=[dict(path=k,sha256=v) for k,v in sorted(pins.items())])
    if a.stage=='replay':request.update(mjb_witness_report=(BASE/'witness/report.json').as_posix(),expected_model_mjb=(BASE/'witness/expected_model.mjb').as_posix())
    assert not request_path.exists() and not folder.exists() and not (BASE/a.stage).exists()
    write_json(request_path,request);folder.mkdir()
    write_new(folder/'run.ps1',run_text(a.stage));write_new(folder/'run_durable.ps1',durable_text(a.stage))
    for path in (request_path,folder/'run.ps1',folder/'run_durable.ps1'):
        pins[path.resolve().as_posix()]=sha(path)
    write_json(folder/'launch_receipt.json',dict(stage=a.stage,request_path=request_path.as_posix(),
        request_sha256=sha(request_path),exact_wsl_arguments=arguments(a.stage),input_hashes=pins,
        requested_native_steps=request['native_step_budget'],requested_serializations=request['serialization_budget'],
        execution_selected=False,final_clearance_required=True,automatic_retry=False,hardware_authorized=False))
    print(json.dumps(dict(request_sha256=sha(request_path),launch_receipt_sha256=sha(folder/'launch_receipt.json'),pins=len(pins),launched=False)))

if __name__=='__main__':main()
