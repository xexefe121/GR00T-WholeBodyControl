"""Pure saved-JSON verdict; preserve raw Python/WSL exit separately."""
import argparse
import json
from pathlib import Path

def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))

def verdict(base,mode,raw):
    base=Path(base);reasons=[]
    if type(raw) is not int or raw!=0:reasons.append('raw_python_exit_unknown_or_nonzero')
    try:
        if mode=='witness':
            r=read(base/'head_witness/report.json')
            if not (r.get('pass_all') is True and r.get('expected_head_calls')==r.get('attempted_head_calls')==r.get('returned_head_calls')==1
                    and r.get('BFM_inference_calls')==r.get('physics_steps')==0 and r.get('fitting_launched') is False):
                reasons.append('one_call_witness_incomplete')
        elif mode=='evaluation':
            r=read(base/'pilot_outcome.json')
            for name,count in (('nominal',1569),('extension',250)):
                s=r.get(name,{})
                if not (s.get('full_segment_completed') is True and s.get('completed_controls')==count and
                        s.get('physics_steps')==count*10 and s.get('failure') is None):
                    reasons.append(name+'_physical_failure_or_incomplete')
                if not ((s.get('quiet_standing_diagnostic') or {}).get('quiet_standing_diagnostic_pass') is True):
                    reasons.append(name+'_quiet_not_passed')
                if s.get('forbidden_inference_calls')!=0:reasons.append(name+'_forbidden_call_or_missing_accounting')
            for name in ('canonical_initial_full291_parity','canonical_prefix250_parity','actual_query250_input_parity','actual_query250_ownexport_output_parity'):
                if read(base/(name+'.json')).get('passed') is not True:reasons.append(name+'_failed')
        else:raise ValueError('invalid mode')
    except Exception as exc:reasons.append('missing_or_invalid_saved_verdict:'+type(exc).__name__+':'+str(exc))
    return dict(kind='direct_target_launch_diagnostic_verdict',passed=not reasons,mode=mode,
        raw_python_exit_code=raw,diagnostic_exit_code=0 if not reasons else 2,reasons=reasons,
        independent_physics_and_intent_audit_required=True,behavioral_qualification=False)

def main():
    p=argparse.ArgumentParser();p.add_argument('--base',type=Path,required=True);p.add_argument('--mode',choices=['witness','evaluation'],required=True)
    p.add_argument('--raw-record',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    r=verdict(a.base,a.mode,read(a.raw_record).get('raw_python_exit_code'))
    with a.output.open('x',encoding='utf-8') as f:json.dump(r,f,indent=2,allow_nan=False);f.write('\n')
    raise SystemExit(r['diagnostic_exit_code'])

if __name__=='__main__':main()
