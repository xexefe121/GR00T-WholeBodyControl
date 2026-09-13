"""Saved-only dispatch metadata correction; preserve all original owner checks."""
import difflib,hashlib,json
from pathlib import Path
BASE=Path(__file__).resolve().parent
OLD=BASE/'verify_completion.py';NEW=BASE/'verify_completion_dispatch_v2.py'
CHECK='''
def validate_dispatch_correction(dispatch):
    original_path=BASE/'dispatch.json';stop_path=BASE/'preflight_dispatch_failure_v1/after_stop.json'
    original_sha='380077859e7ffc6b3c9ce779cb1d425be0a02aa2e354bb682fe8970ffdc4c750'
    stop_sha='90b305d2fc6b80f2a6f1b258bafacb25ae00c089f3beb65ad6a7cb1f8477c490'
    assert sha(original_path)==original_sha==dispatch['original_dispatch_sha256']
    assert sha(stop_path)==stop_sha==dispatch['preflight_after_stop_sha256']
    original=read(original_path);stop=read(stop_path)
    assert original['wrapper_pid']==27700 and original['handle_acquired'] is True
    assert stop['stopped_only_confirmed_owned_wrapper']==27700 and stop['handle_acquired'] is True
    assert stop['exit_code_known'] is True and stop['observed_wrapper_exit_code']==-1
    assert stop['observed_present_pids']==[] and stop['process_directory_absent'] is True and stop['results_directory_absent'] is True
    assert stop['actual_saved_audit_calls']==0 and stop['automatic_retry'] is False
    assert dispatch['corrected_argument_only_selection'] is True and dispatch['prior_actual_audit_calls']==0 and dispatch['automatic_retry'] is False
    assert dispatch['wrapper_pid']!=original['wrapper_pid'] and dispatch['handle_acquired'] is True
    args=dispatch['exact_arguments']
    assert len(args)==8 and args[:5]==['-NoProfile','-NonInteractive','-ExecutionPolicy','Bypass','-File']
    assert Path(args[5]).resolve()==(BASE/'run_audit_durable.ps1').resolve()
    assert args[6:] == ['-LaunchReceiptSha256',sha(BASE/'launch_receipt.json')]
    assert original['launch_receipt_sha256']==dispatch['launch_receipt_sha256']==sha(BASE/'launch_receipt.json')
    assert original['request_sha256']==dispatch['request_sha256']==sha(BASE/'request.json')
    assert original['clearance_sha256']==dispatch['clearance_sha256']==sha(BASE/'launch_clearance.json')
    assert sha(BASE/'preflight_dispatch_failure_v1/dispatch.json')==original_sha
    assert datetime.fromisoformat(dispatch['utc'].replace('Z','+00:00'))>datetime.fromisoformat(stop['utc'].replace('Z','+00:00'))
    return dict(original_dispatch_sha256=original_sha,preflight_after_stop_sha256=stop_sha,
        corrected_dispatch_sha256=sha(BASE/'dispatch_v2.json'),prior_actual_audit_calls=0,
        original_wrapper_forcibly_stopped_before_script_body=True,argument_only_dispatch_correction=True)

'''
old=OLD.read_text()
assert old.count("BASE/'dispatch.json'")==2
text=old.replace("BASE/'dispatch.json'","BASE/'dispatch_v2.json'")
assert text.count('def main():')==1
text=text.replace('def main():',CHECK+'def main():')
marker="    assert start['wrapper_pid']==child['wrapper_pid']==end['wrapper_pid']==dispatch['wrapper_pid']"
assert text.count(marker)==1;text=text.replace(marker,"    correction=validate_dispatch_correction(dispatch)\n"+marker)
marker="    output_paths+=list((BASE/'process_v1').glob('*.json'))"
assert text.count(marker)==1
text=text.replace(marker,"    output_paths += [BASE/'dispatch.json',BASE/'verify_completion.py',BASE/'derive_owner_dispatch_v2.py',BASE/'owner_dispatch_v2.diff'] + list((BASE/'preflight_dispatch_failure_v1').glob('*'))\n"+marker)
marker='    result=dict(completion_accounting_passed=True'
assert text.count(marker)==1;text=text.replace(marker,'    result=dict(dispatch_correction=correction,completion_accounting_passed=True')
text=text.replace("BASE/'owner_completion.json'","BASE/'owner_completion_dispatch_v2.json'")
with NEW.open('x',encoding='utf-8') as f:f.write(text)
with (BASE/'owner_dispatch_v2.diff').open('x') as f:f.write(''.join(difflib.unified_diff(old.splitlines(True),text.splitlines(True),fromfile='preserved/verify_completion.py',tofile='verify_completion_dispatch_v2.py')))
print(json.dumps(dict(source_sha256=hashlib.sha256(NEW.read_bytes()).hexdigest(),original_sha256=hashlib.sha256(OLD.read_bytes()).hexdigest(),executed=False)))
