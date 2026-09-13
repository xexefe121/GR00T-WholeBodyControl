"""Saved232-case verdicts and saved proposal arithmetic only, no real inference."""
from pathlib import Path
from types import SimpleNamespace
import hashlib
import json
import sys
import numpy as np
BASE=Path(__file__).resolve().parent.parent;NEW=BASE.parent
sys.path.insert(0,str(BASE/'source_draft_v1'))
from ordered_admission import admit
import transactional_student as module
from gear_sonic.utils.g1_true23_bfm_seed_observations import BFMHistory,state_and_terms
def load(path):
    with np.load(path,allow_pickle=False) as a:return {k:a[k].copy() for k in a.files}
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    source=NEW/'phase_student_preallocated_forecast_v2';reportpath=source/'results/report.json'
    report=json.loads(reportpath.read_text());assert report['pass_'] and report['cases']==232
    cases=load(source/'cases.npz');names=('recorded_student','recorded_original_bfm_clipped','previous_actual_target','current_joint_pose')
    results=[]
    for control in range(250,279):
        ids=[int(np.flatnonzero((cases['control']==control)&(cases['candidate']==name)&(cases['horizon_controls']==5))[0]) for name in names]
        targets=[cases['target'][index] for index in ids];verdicts=[report['rows'][index] for index in ids]
        for i in range(4):
            for j in range(i):
                if np.array_equal(targets[i],targets[j]):assert verdicts[i]['feasible']==verdicts[j]['feasible']
        calls=[]
        def replay(target):
            index=next(i for i,value in enumerate(targets) if np.array_equal(value,target));calls.append(index)
            return dict(feasible=bool(verdicts[index]['feasible']),first_failure=verdicts[index]['first_failure'],saved_fixture_only=True)
        selected,target,ledger=admit(targets,replay)
        wanted=next((index for index,item in enumerate(verdicts) if item['feasible']),None)
        assert selected==wanted
        if selected is not None:np.testing.assert_array_equal(target,targets[selected])
        results.append(dict(control=control,selected_from_saved_verdicts=selected,forecasts_called=len(calls),records=len(ledger)))
    actualpath=NEW/'fast_controller_phase_fit_v1/nominal/trace.npz';actual=load(actualpath)
    histpath=NEW/'phase_student_failure_diagnosis_v1/arrays.npz';histories=load(histpath)
    contractpath=Path('Z:/codex/GR00T-WholeBodyControl-sonic-transfer-23dof/artifacts/teleop_six_hour_20260910/mjbatch_native23_inputs_v1/contract.json')
    c=json.loads(contractpath.read_text());default,kp,effort=[np.asarray(c[k]) for k in ('default_q','kp','training_effort')]
    proposals=[]
    for control in range(250,253):
        h=BFMHistory()
        for key in h.data:h.data[key][:]=histories['actual_history_'+key][control-250]
        seed=SimpleNamespace(history=h,previous_action=actual['previous_action'][control].copy(),recorded_controls=control)
        seed._terms=lambda q,v,a:state_and_terms(q[7:],v[6:],q[3:7],v[3:6],a,default)
        runtime=SimpleNamespace(seed=seed,c=c,limits=np.asarray(c['joint_limits']),
            features=lambda *args:actual['features'][control].copy(),
            head=SimpleNamespace(run=lambda *args:[actual['delta'][control][None].copy()]))
        raw=((actual['base_target'][control]-default)*kp/(.25*effort)).astype(np.float32)
        module.infer_base=lambda *args:(raw.copy(),actual['base_target'][control].copy(),actual['state'][control].copy())
        transaction=module.TransactionalStudent(runtime)
        proposed=transaction.begin(control,actual['qpos'][control],actual['qvel'][control])
        for key in ('target','delta','base_target','features','history','state','previous_action','action'):
            np.testing.assert_array_equal(proposed[key],actual[key][control])
        result=transaction.commit(proposed['target'])
        expected=((result['target']-default)*kp/(.25*effort)).astype(np.float32)
        np.testing.assert_array_equal(result['action'],expected)
        proposals.append(dict(control=control,proposal_arithmetic_bitexact_to_saved65000=True,
            selected_action_bitexact_to_expert_formula=True,action_matches_old_raw_semantics=bool(np.array_equal(result['action'],actual['action'][control]))))
    result=dict(pass_=True,saved_case_selection=results,recorded_proposal_arithmetic=proposals,
        actual_inference_calls=0,private_forecasts=0,physics_steps=0,optimizer_calls=0,
        saved_fixture_selection_is_not_a_new_connected_rollout=True,
        input_sha256={str(p):sha(p) for p in (source/'cases.npz',reportpath,actualpath,histpath,contractpath,Path(__file__))})
    (BASE/'recorded_fixture_test_report.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(pass_=True,saved_controls_checked=29,recorded_proposal_arithmetic=proposals,actual_inference_calls=0,physics_steps=0)))

if __name__=='__main__':main()
