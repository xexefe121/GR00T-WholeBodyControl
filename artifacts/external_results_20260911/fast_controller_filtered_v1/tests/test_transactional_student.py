"""Stub-only transactional invariants; no actual model inference or physics."""
from pathlib import Path
from types import SimpleNamespace
import copy
import json
import sys
import numpy as np
BASE=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(BASE/'source_draft_v1'))
import transactional_student as module
from gear_sonic.utils.g1_true23_bfm_seed_observations import BFMHistory

def equal(seed,snapshot):
    assert seed.recorded_controls==snapshot[0]
    np.testing.assert_array_equal(seed.previous_action,snapshot[1])
    for key in seed.history.data:np.testing.assert_array_equal(seed.history.data[key],snapshot[2][key])

def snapshot(seed):return seed.recorded_controls,seed.previous_action.copy(),copy.deepcopy(seed.history.data)

def make(control):
    history=BFMHistory()
    for index,(key,value) in enumerate(history.data.items()):value[:]=index+.125
    seed=SimpleNamespace(recorded_controls=control,previous_action=np.arange(23,dtype=np.float32)/10,history=history)
    seed._terms=lambda q,v,a:(np.zeros(52,np.float32),{k:np.full(value.shape[1],7,np.float32) for k,value in history.data.items()})
    c=dict(default_q=np.zeros(23),kp=np.full(23,4.),training_effort=np.full(23,4.))
    features=lambda q,v,f,b,a:np.r_[np.full(1046,3,np.float32),a].astype(np.float32)
    head=SimpleNamespace(run=lambda *_:[np.full((1,23),.75,np.float32)])
    runtime=SimpleNamespace(seed=seed,c=c,limits=np.tile([-1.,1.],(23,1)),features=features,head=head)
    return module.TransactionalStudent(runtime)

def main():
    # Raw normalized action8 requests target2, while native target clips to1.
    calls=[]
    def infer(seed,q,v,a,h,frame,terminal):
        calls.append((frame,terminal));return np.full(23,8,np.float32),np.full(23,2.),np.zeros(52,np.float32)
    module.infer_base=infer
    q=np.zeros(30);v=np.zeros(29);checks=[]
    for control,terminal,disabled in ((0,False,True),(249,False,True),(250,False,False),(1269,True,False)):
        transaction=make(control);seed=transaction.runtime.seed;before=snapshot(seed)
        proposed=transaction.begin(control,q,v,terminal,disabled);equal(seed,before)
        assert transaction.pending is not None
        try:transaction.begin(control,q,v,terminal,disabled)
        except RuntimeError:pass
        else:raise AssertionError('Second proposal accepted before commit.')
        transaction.cancel();equal(seed,before)
        proposed=transaction.begin(control,q,v,terminal,disabled);equal(seed,before)
        result=transaction.commit(proposed['target'])
        assert seed.recorded_controls==control+1 and transaction.pending is None
        wanted=8. if control<250 or terminal else 4.
        np.testing.assert_array_equal(seed.previous_action,np.full(23,wanted,np.float32))
        np.testing.assert_array_equal(result['proposed_action'],np.full(23,8. if disabled or terminal else 11.,np.float32))
        np.testing.assert_array_equal(result['target'],np.ones(23))
        for key in seed.history.data:np.testing.assert_array_equal(seed.history.data[key][0],np.full(seed.history.data[key].shape[1],7,np.float32))
        try:transaction.commit(proposed['target'])
        except RuntimeError:pass
        else:raise AssertionError('Repeated commit accepted.')
        checks.append(dict(control=control,terminal=terminal,unchanged_until_commit=True,history_advanced_once=True,action_semantics_exact=True))
    for control,terminal in ((250,False),(1269,True)):
        transaction=make(control);seed=transaction.runtime.seed;before=snapshot(seed)
        transaction.begin(control,q,v,terminal,False)
        try:transaction.commit(np.full(23,2.))
        except ValueError:pass
        else:raise AssertionError('Out-of-range target accepted.')
        equal(seed,before)
        result=transaction.commit(np.full(23,.125))
        np.testing.assert_array_equal(seed.previous_action,np.full(23,.5,np.float32))
        assert not result['primary_applied_unchanged']
        checks.append(dict(control=control,fallback_records_applied_target=True,invalid_commit_preserves_history=True))
    transaction=make(250);transaction.begin(250,q,v);transaction.runtime.seed.previous_action[0]+=1
    changed=snapshot(transaction.runtime.seed)
    try:transaction.commit(np.zeros(23))
    except RuntimeError:pass
    else:raise AssertionError('Stale proposal accepted.')
    equal(transaction.runtime.seed,changed)
    transaction=make(250);before=snapshot(transaction.runtime.seed);count_before=len(calls)
    try:transaction.begin(250,q,v,terminal=True)
    except ValueError:pass
    else:raise AssertionError('Wrong phase flags accepted.')
    equal(transaction.runtime.seed,before);assert len(calls)==count_before
    transaction.runtime.head.run=lambda *_:[np.full((1,23),np.nan,np.float32)]
    try:transaction.begin(250,q,v)
    except ValueError:pass
    else:raise AssertionError('Nonfinite head output accepted.')
    equal(transaction.runtime.seed,before);assert transaction.pending is None
    np.testing.assert_array_equal(q,np.zeros(30));np.testing.assert_array_equal(v,np.zeros(29))
    result=dict(pass_=True,checks=checks,stale_commit_rejected_without_overwriting_state=True,
        caller_qpos_qvel_unchanged=True,phase_error_rejected_before_inference=True,nonfinite_proposal_preserves_state=True,
        stub_inference_calls=len(calls),actual_inference_calls=0,physics_steps=0,optimizer_calls=0)
    (BASE/'transactional_stub_test_report_v2.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result))

if __name__=='__main__':main()
