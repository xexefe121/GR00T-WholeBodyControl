"""Versioned diagnostic/reporting fixes only; preserve all v1 source/evidence."""
from pathlib import Path
import hashlib,json,shutil
BASE=Path(__file__).resolve().parent
SRC=BASE/'source_draft_v1';DEST=BASE/'source_draft_v2'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
if DEST.exists():
    assert not (BASE/'preservation_v2_derivation.json').exists()
    assert all(sha(p)==sha(SRC/p.relative_to(DEST)) for p in DEST.rglob('*.py'))
else:DEST.mkdir()
for p in SRC.rglob('*.py'):
    q=DEST/p.relative_to(SRC);q.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,q)
path=DEST/'physical_fit_diagnostics.py';text=path.read_text(encoding='utf-8');edits=[]
def replace(old,new,count=1):
    global text
    assert text.count(old)==count,(old[:60],text.count(old),count)
    text=text.replace(old,new);edits.append(dict(old=old,new=new,count=count))
replace('    last_outputs = {}\n', '    last_outputs = {}\n    current_call_returned=False\n    current_stage="not_started"\n    current_batch=np.array([-1,-1],np.int64)\n')
replace('                current_input=features[start:stop].copy()\n', '''                current_input=features[start:stop].copy()
                current_stage='Torch';current_batch=np.array([start,stop],np.int64)
                current_call_returned=False;last_output=np.empty((0,23),np.float32);last_outputs={}
''')
replace("                ledger['diagnostic_torch_rows_returned']+=stop-start", "                ledger['diagnostic_torch_rows_returned']+=stop-start\n                current_call_returned=True")
replace('        if session_factory is None:', '''        current_stage='construct_ORT';current_batch=np.array([-1,-1],np.int64)
        current_input=np.empty((0,1069),np.float32)
        current_call_returned=False;last_output=np.empty((0,23),np.float32);last_outputs={}
        if session_factory is None:''')
replace('\n            current_input=features[start:stop].copy()\n', '''
            current_input=features[start:stop].copy()
            current_stage='ORT';current_batch=np.array([start,stop],np.int64)
            current_call_returned=False;last_output=np.empty((0,23),np.float32);last_outputs={}
''')
replace("            ledger['head_onnx_calls_returned']+=1", "            ledger['head_onnx_calls_returned']+=1\n            current_call_returned=True")
replace('            current_input=current_input,last_returned_output=last_output,**last_outputs)', '''            current_input=current_input,last_returned_output=last_output,**last_outputs,
            current_call_returned=np.asarray(current_call_returned),current_stage=np.asarray(current_stage),
            current_batch=current_batch)''')
replace("                first24_response_MSE=float(np.mean(pair[ids[:24]]**2)),\n",'')
replace('        cells.append(item)', '''        first=(251,350,1169)[len(cells)%3]
        window=ids[(physical['successor_control'][ids]>=first)&(physical['successor_control'][ids]<first+24)]
        item['first24_requested_window']=dict(successor_control_range=[first,first+23],requested=24,
            valid=len(window),strict_failed=24-len(window),
            response_valid_MSE=float(np.mean(pair[window]**2)) if len(window) else None)
        cells.append(item)''')
path.write_text(text,encoding='utf-8')
tests=(BASE/'test_preparation.py').read_text(encoding='utf-8').replace("SOURCE=BASE/'source_draft_v1'","SOURCE=BASE/'source_draft_v2'")
tests+='''
def test_stub_later_batch_fault_has_no_stale_output(tmp_path):
    class LaterFault:
        calls=0
        def run(self,_,feed):
            self.calls+=1
            if self.calls==2:raise RuntimeError('second synthetic call fault')
            return [np.zeros((len(feed['features']),23),np.float32)]
    ledger=stub_ledger()
    with pytest.raises(RuntimeError):
        evaluate_physical(stub_actor,np.zeros((257,1069),np.float32),np.zeros(1069,np.float32),np.ones(1069,np.float32),np.ones(23,np.float32),
            'no-model.onnx',ledger,tmp_path,'later',4,session_factory=lambda _:LaterFault())
    assert ledger['head_onnx_calls_attempted']==2 and ledger['head_onnx_calls_returned']==1
    with np.load(tmp_path/'later_failed_physical_diagnostics.npz') as z:
        assert z['current_input'].shape==(1,1069) and z['last_returned_output'].shape==(0,23)
        assert not bool(z['current_call_returned']) and str(z['current_stage'])=='ORT'
        assert np.array_equal(z['current_batch'],[256,257]) and int(z['valid_onnx'].sum())==256

def test_fixed_requested_window_keeps_failed_early_clocks():
    from physical_fit_diagnostics import physical_metrics
    cells=[np.array([0,1],np.int64)]+[np.empty(0,np.int64) for _ in range(8)]
    centers=dict(base_target=np.zeros((3,23)),expert_target=np.zeros((3,23)),joint_limits=np.tile([-10.,10.],(23,1)))
    physical=dict(successor=np.array([1,2]),successor_control=np.array([252,290]),base=np.zeros((2,23)),target=np.zeros((2,23)),
        cells=cells,coverage=[dict(dataset=i//3,phase=('acquisition','source','return')[i%3],requested=(99,819,100)[i%3],
        valid=len(ids),strict_failed=(99,819,100)[i%3]-len(ids),empty=len(ids)==0) for i,ids in enumerate(cells)])
    for key in ('policy_native_target_clipped','teacher_feedback_clipped','teacher_native_clipped','successor_replan_boundary','successor_zero_gain'):
        physical[key]=np.zeros(2,bool)
    result=physical_metrics(np.zeros((3,23)),np.stack([np.ones(23),np.full(23,4.)]),centers,physical,np.ones(23))
    window=result['cells'][0]['first24_requested_window']
    assert window==dict(successor_control_range=[251,274],requested=24,valid=1,strict_failed=23,response_valid_MSE=1.)
'''
(BASE/'test_preparation_v2.py').write_text(tests,encoding='utf-8')
report=dict(preparation_only=True,actual_model_evaluations=0,optimizer_updates=0,physics_steps=0,
    prior_preparation_report_sha256=sha(BASE/'preparation_report.json'),
    unchanged_trainer_sha256=sha(DEST/'fit_physical_continuation.py'),
    modified_file='physical_fit_diagnostics.py',old_sha256=sha(SRC/'physical_fit_diagnostics.py'),new_sha256=sha(path),exact_substitutions=edits,
    source_sha256={p.relative_to(DEST).as_posix():sha(p) for p in DEST.rglob('*.py')})
(BASE/'preservation_v2_derivation.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
print(json.dumps(dict(copied_sources=len(report['source_sha256']),modified_files=1,diagnostics_sha256=sha(path))))
