"""Pure callback tests: no private forecast implementation or dynamics executed."""
from pathlib import Path
import json
import sys
import numpy as np
BASE=Path(__file__).resolve().parent.parent;sys.path.insert(0,str(BASE/'source_draft_v1'))
from ordered_admission import admit

def main():
    targets=[np.full(23,value) for value in (0.,0.,.5,-.5)];calls=[]
    def forecast(target):
        calls.append(target.copy());return dict(feasible=bool(target[0]==.5),first_failure=None if target[0]==.5 else 'stub_rejection')
    index,target,records=admit(targets,forecast)
    assert index==2 and len(calls)==2 and len(records)==3 and records[1]['duplicate_of']==0
    np.testing.assert_array_equal(target,targets[2]);target[:]=17
    np.testing.assert_array_equal(targets[2],np.full(23,.5))
    count=[0]
    def reject(target):count[0]+=1;return dict(feasible=False,first_failure='stub_rejection')
    index,target,records=admit(targets,reject)
    assert index is None and target is None and count[0]==3 and len(records)==4
    called=[0]
    def accept(target):called[0]+=1;return dict(feasible=True)
    index,target,records=admit(targets,accept)
    assert index==0 and called[0]==1 and len(records)==1
    try:admit(targets,lambda _:dict(feasible=np.bool_(True)))
    except ValueError:pass
    else:raise AssertionError('Non-Python Boolean forecast accepted.')
    result=dict(pass_=True,fixed_first_feasible_order=True,duplicate_identity_recorded=True,exact_duplicates_forecast_once=True,
        reject_all_returns_no_target=True,candidate_inputs_unchanged=True,actual_forecasts=0,physics_steps=0,inference_calls=0)
    (BASE/'ordered_admission_stub_test_report.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result))

if __name__=='__main__':main()
