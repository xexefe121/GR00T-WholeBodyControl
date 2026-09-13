"""Read-only qualification, continuity and native-grid visualization contract."""
import hashlib
import json
from pathlib import Path
import numpy as np

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):
    with np.load(path,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}
def exact(a,b,name):
    a,b=np.asarray(a),np.asarray(b)
    if a.shape!=b.shape or a.dtype!=b.dtype or a.tobytes()!=b.tobytes():raise ValueError('Exact mismatch: '+name)

def load_qualified(path,digest):
    path=Path(path)
    if sha(path)!=digest:raise ValueError('Qualification digest changed')
    q=json.loads(path.read_text())
    assert (q['lifecycle_controls'],q['source_controls'],q['separate_hold_controls'],q['physics_steps'])==(1417,667,250,16670)
    assert q['both_quiet_windows_pass'] and not q['timing_qualified'] and not q['live_teleoperation_qualified']
    pins={str(path):digest}
    for entry in [*q['traces'].values(),*q['independent_reports'].values()]:
        p=Path(entry['path']);assert sha(p)==entry['sha256'];pins[str(p)]=sha(p)
    for part,count,start in (('full',1417,0),('hold',250,1417)):
        physics=json.loads(Path(q['independent_reports'][part+'_physics']['path']).read_text())
        intent=json.loads(Path(q['independent_reports'][part+'_intent']['path']).read_text())
        trace_sha=q['traces'][part]['sha256']
        assert physics['independent_segment_pass'] and physics['recorded_trace_reproduced_through_last_sample']
        assert physics['compared_physics_steps']==count*10 and physics['intended_segment_controls']==count
        assert all(physics['original_trace_comparison'].values()) and trace_sha in physics['input_hashes'].values()
        assert intent['clip']=='walk002' and intent['global_start']==start and intent['requested_controls']==count
        assert intent['independent_physical_pass'] and intent['intended_segment_completed'] and intent['requested_segment_quiet_pass']
        assert trace_sha in intent['hashes'].values() and q['independent_reports'][part+'_physics']['sha256'] in intent['hashes'].values()
        assert all(intent['quiet_last_three_seconds']['gates'].values())
        if part=='full':
            assert intent['full_lifecycle_source_intent_pass'] and intent['source_metrics']['source_controls']==667
            assert all(intent['source_metric_gates'].values())
    full,hold=[read(q['traces'][part]['path']) for part in ('full','hold')]
    for a,start,count in ((full,0,1417),(hold,1417,250)):
        assert a['physics_qpos'].shape==(count*10+1,30) and a['physics_qvel'].shape==(count*10+1,29)
        assert a['physics_torque'].shape==(count*10,23) and np.all(a['physics_substeps']==10)
        exact(a['global_control'],np.arange(start,start+count,dtype=a['global_control'].dtype),'global controls')
        exact(a['source_frame'],np.minimum(np.arange(start,start+count)+11,1427).astype(a['source_frame'].dtype),'source frames')
        exact(a['physics_time'],a['physics_expected_time'],'independent repeated clock')
    assert full['final_integration'].shape==hold['initial_integration'].shape==(291,)
    for key in full:
        if key.startswith('final_') and 'initial_'+key[6:] in hold:
            exact(full[key],hold['initial_'+key[6:]],'full/hold '+key)
    for key in ('physics_qpos','physics_qvel','physics_time','physics_warning_number','physics_warning_lastinfo'):
        exact(full[key][-1],hold[key][0],'main/hold native boundary '+key)
    return q,pins,full,hold

CONTACT_STEPS=(0,3500,8000,10170,11170,12650,14170,16670)
def visual_grid():
    steps=np.unique(np.r_[np.arange(0,16671,50),CONTACT_STEPS])
    frames=np.minimum(np.where(steps==0,10,(steps-1)//10+11),1427)
    durations=np.r_[np.diff(steps)*.002,.002]
    assert steps[0]==0 and steps[-1]==16670 and np.all(durations>0)
    return steps,frames,durations
