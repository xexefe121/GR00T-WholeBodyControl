"""Read-only accounting predicate; importing this module invokes no native code."""
import argparse,json
from pathlib import Path

def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))

def validate_report(stage,r):
    assert r['passed'] is True and r['error'] is None
    assert r['model_inference_calls']==r['optimizer_updates']==r['other_oracle_native_steps']==0
    assert r['plant_foundation_connected'] is False and r['real_wallclock_experiment'] is False
    c=r['api_counters'];steps,saves=(0,2) if stage=='witness' else (21348,8)
    assert c['step_attempted']==c['step_returned']==steps
    assert c['serialization_attempted']==c['serialization_returned']==saves
    assert c['denied_step_calls']==c['denied_serialization_calls']==0
    assert len(c['serialization_records'])==saves
    assert all(x['native_returned'] is True and x.get('error') is None and x.get('capture_error') is None for x in c['serialization_records'])
    if stage=='replay':
        for key,n,verified in (('expert',18190,18190),('direct',3158,3157)):
            case=r[key];assert case['equivalence_passed'] is True and case['error'] is None
            counts=case['adapter_counters']
            assert all(counts[k]==n for k in ('attempted','returned','capture_attempts','captured','verification_attempts'))
            assert counts['verified']==verified
        assert r['original_direct_physics_pass'] is False and r['exact_expected_failure_reproduction'] is True
    return True

def main():
    p=argparse.ArgumentParser();p.add_argument('--base',type=Path,required=True)
    p.add_argument('--stage',choices=['witness','replay'],required=True);a=p.parse_args()
    folder=a.base/(a.stage+'_process');raw=read(folder/'raw_exit.json');passed=False;error=None
    try:
        assert raw['known'] is True and raw['raw_python_exit_code']==0 and raw['raw_error'] is None
        validate_report(a.stage,read(a.base/a.stage/'report.json'));passed=True
    except BaseException as exc:error=repr(exc)
    result=dict(passed=passed,raw_python_exit_code=raw['raw_python_exit_code'],diagnostic_exit_code=0 if passed else 2,error=error)
    with (folder/'diagnostic_verdict.json').open('x',encoding='utf-8') as f:json.dump(result,f,indent=2);f.write('\n')
    raise SystemExit(result['diagnostic_exit_code'])

if __name__=='__main__':main()
