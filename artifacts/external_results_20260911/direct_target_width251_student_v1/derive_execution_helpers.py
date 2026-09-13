"""Preserve qualified wrapper mechanics; adapt metadata and add bound launch receipt."""
from pathlib import Path
import difflib
BASE=Path(__file__).resolve().parent
OLD=BASE.parent/'direct_target_causal_width512_student_v1'

def derive():
    original=(OLD/'run_fit_durable_v1.ps1').read_text(encoding='utf-8-sig')
    text=original.replace("param([Parameter(Mandatory=$true)][ValidatePattern('^[0-9a-f]{64}$')][string]$ClearanceSha256)","param([Parameter(Mandatory=$true)][ValidatePattern('^[0-9a-f]{64}$')][string]$ClearanceSha256,\n      [Parameter(Mandatory=$true)][ValidatePattern('^[0-9a-f]{64}$')][string]$LaunchReceiptSha256)")
    for a,b in [('causal_width512_warm_continuation','qualified_width251_recovery_warm512_fit'),('train_response_balanced.py','train_recovery.py'),('ordinary_start_step=71000','ordinary_start_step=81000'),('ordinary_final_step=81000','ordinary_final_step=91000'),('optimizer_start_step=6000','optimizer_start_step=16000'),('optimizer_final_step=16000','optimizer_final_step=26000'),('ordinary_start_step -ne 71000','ordinary_start_step -ne 81000'),('ordinary_final_step -ne 81000','ordinary_final_step -ne 91000'),('optimizer_start_step -ne 6000','optimizer_start_step -ne 16000'),('optimizer_final_step -ne 16000','optimizer_final_step -ne 26000'),('optimizer_step -ne 16000','optimizer_step -ne 26000'),('146860000','157040000'),('30000','40000'),('367570','368588'),('1437','1441')]:text=text.replace(a,b)
    text=text.replace("$request.coefficient -ne 1.8188207859141674d)","$request.coefficient -ne 1.8188207859141674d -or $request.recovery_coefficient -ne 0.2d -or $request.expansion_performed -ne $false -or $request.recovery_rows -ne 1018 -or $request.learning_rate_values.Count -ne 10000)")
    text=text.replace("$report.expansion_seed -ne 20260912","$report.expansion_performed -ne $false -or $report.recovery_rows -ne 1018 -or $report.recovery_coefficient -ne 0.2d")
    marker="    $clear=Read-SharedJson $clearancePath\n"
    addition="""    $launchReceiptPath=Join-Path $runRoot 'launch_receipt.json'
    if((Read-SharedSha $launchReceiptPath) -ne $LaunchReceiptSha256){throw 'Launch receipt identity differs.'}
    $launchReceipt=Read-SharedJson $launchReceiptPath
    if($clear.launch_receipt_sha256 -ne $LaunchReceiptSha256 -or $launchReceipt.request_sha256 -ne $clear.request_sha256 -or $launchReceipt.frozen_receipt_sha256 -ne $clear.frozen_receipt_sha256 -or $launchReceipt.launcher_sha256 -ne $clear.launcher_sha256 -or $launchReceipt.automatic_retry -ne $false){throw 'Launch receipt not bound by clearance.'}
"""
    assert text.count(marker)==1;text=text.replace(marker,marker+addition)
    text=text.replace("$review.frozen_receipt_sha256 -ne $clear.frozen_receipt_sha256)","$review.frozen_receipt_sha256 -ne $clear.frozen_receipt_sha256 -or $review.launch_receipt_sha256 -ne $LaunchReceiptSha256 -or $review.launcher_sha256 -ne $clear.launcher_sha256)")
    marker="    $pins[$receiptPath]=$clear.frozen_receipt_sha256;$pins[$requestPath]=$clear.request_sha256\n"
    addition="""    $expectedPins=[ordered]@{}
    foreach($key in $pins.Keys){
        $canonical=[IO.Path]::GetFullPath($key).Replace('\\','/').ToLowerInvariant()
        if($expectedPins.Contains($canonical)){throw 'Duplicate frozen physical pin.'}
        $expectedPins[$canonical]=$pins[$key]
    }
    $actualPins=[ordered]@{}
    foreach($property in $launchReceipt.input_sha256.PSObject.Properties){
        $canonical=[IO.Path]::GetFullPath($property.Name).Replace('\\','/').ToLowerInvariant()
        if($actualPins.Contains($canonical)){throw 'Duplicate launch physical pin.'}
        $actualPins[$canonical]=$property.Value
    }
    if($expectedPins.Count -ne $actualPins.Count -or $launchReceipt.pin_count -ne $actualPins.Count){throw 'Launch pin membership count differs.'}
    foreach($key in $expectedPins.Keys){if(-not $actualPins.Contains($key) -or $actualPins[$key] -ne $expectedPins[$key]){throw 'Launch pin membership differs.'}}
    $pins[$launchReceiptPath]=$LaunchReceiptSha256
"""
    assert text.count(marker)==1;text=text.replace(marker,marker+addition)
    text=text.replace('clearance_sha256=$ClearanceSha256;', 'clearance_sha256=$ClearanceSha256;launch_receipt_sha256=$LaunchReceiptSha256;')
    text=text.replace("'postrun_pins.json'))", "'postrun_pins.json','process_absence.json'))")
    (BASE/'run_fit_durable_v1.ps1').write_text(text,encoding='utf-8',newline='\n')
    (BASE/'launcher_derivation.patch').write_text(''.join(difflib.unified_diff(original.splitlines(True),text.splitlines(True),fromfile='qualified_width512_launcher',tofile='recovery_warm512_launcher')),encoding='utf-8')
    owner=(OLD/'verify_completed.py').read_text(encoding='utf-8')
    # Keep every general hash, pin, exit and bounded counter primitive unchanged.
    begin=owner[:owner.index('def validate_report(report):')]
    (BASE/'execution_common.py').write_text(begin.replace('BASE=Path(__file__).resolve().parent\n',''),encoding='utf-8',newline='\n')
    (BASE/'read_fit_progress_v1.ps1').write_bytes((OLD/'read_fit_progress_v1.ps1').read_bytes())

if __name__=='__main__':derive()
