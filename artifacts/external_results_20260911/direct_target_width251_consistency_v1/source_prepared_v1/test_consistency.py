"""Synthetic numeric fixtures only; no task arrays, checkpoints, or runtimes."""
import ast
import copy
import hashlib
import json
from pathlib import Path
import numpy as np
import pytest
import admission as gate
import saved_inputs as loader
from consistency_math import alias_groups,proximity,fixed_queries,OLD,TOTAL,NEW,MODES
from normalization_math import normalize64,verify_export_algebra
from direct_contract import RETAINED,normalized_labels
from input_schema import load_numeric

EXPORT=Path(__file__).resolve().parents[2]/'direct_target_causal_width512_student_v1/source_prepared_v1/width512_promoted.py'

def fixture(n=4):
    return np.zeros((n,1323),np.float32),np.zeros((n,23),np.float64),np.zeros(1323,np.float32),np.ones(1323,np.float32)

def group(x,y,m,s,mode='raw1323_bits',**kw):
    return alias_groups(x,y,y.astype(np.float32),m,s,mode,**kw)

def test_exact_alias_conflict_members_and_anchor_deltas_retained():
    x,y,m,s=fixture();x[2,0]=1;y[1,0]=.25;y[3,1]=.5
    arrays,summary=group(x,y,m,s,old_rows=2)
    assert arrays['members'].tolist()==[0,1,3] and arrays['offsets'].tolist()==[0,3]
    assert arrays['group_id'].tolist()==[0,0,-1,0]
    assert summary['new_old_groups']==summary['new_raw_radian_disagreement_groups']==1
    np.testing.assert_array_equal(arrays['target_delta_from_anchor'][[0,1,3]],y[[0,1,3]])
    assert arrays['target_RMSE_from_anchor'][1]==np.sqrt(.25**2/23)
    assert arrays['target_max_abs_from_anchor'][3]==.5
    assert arrays['target_max'][0,:2].tolist()==[.25,.5]
    np.testing.assert_array_equal(x,np.r_[np.zeros((2,1323),np.float32),x[2:]])

def test_prior_and_history_resolve_current_only_aliases():
    x,y,m,s=fixture(3);x[1,1000]=1;x[2,1023]=1
    assert group(x,y,m,s)[1]['duplicate_groups']==0
    arrays,summary=group(x,y,m,s,'current1000_bits')
    assert summary['duplicate_groups']==1 and arrays['distinct_raw1323_count'].tolist()==[3]

@pytest.mark.parametrize('prefix',['raw1323','current1000','normalized64'])
def test_signed_zero_numeric_equivalence_is_separate(prefix):
    x,y,m,s=fixture(2);x[1,0]=-0.
    assert x[0].tobytes()!=x[1].tobytes()
    assert group(x,y,m,s,prefix+'_bits')[1]['duplicate_groups']==0
    assert group(x,y,m,s,prefix+'_zero_equivalent')[1]['duplicate_groups']==1

def test_distinct_raw_inputs_can_collapse_in_float64_centering():
    x,y,m,s=fixture(2);x[:,0]=[1,2];m[0]=np.float32(1e30)
    assert group(x,y,m,s)[1]['duplicate_groups']==0
    a,r=group(x,y,m,s,'normalized64_bits')
    assert r['collapsed_distinct_raw1323_groups']==1 and a['members'].tolist()==[0,1]

def test_raw_target_disagreement_casts_to_identical_labels():
    x,y,m,s=fixture(2);y[:]=1;y[1,0]=np.nextafter(1.,2.)
    _,r=group(x,y,m,s)
    assert r['raw_radian_disagreement_groups']==1 and r['normalized_label_disagreement_groups']==0

def test_hash_collision_rejected_not_alias_admitted():
    x,y,m,s=fixture(2);x[1,0]=1
    class BadHash:
        def hexdigest(self):return '0'*64
    with pytest.raises(ValueError,match='Digest collision'):group(x,y,m,s,digest=lambda _:BadHash())

def test_alias_across_256_row_boundary():
    x,y,m,s=fixture(258);x[:,0]=np.arange(258);x[257]=x[0];y[257,4]=.125
    a,r=group(x,y,m,s)
    assert a['members'].tolist()==[0,257] and r['duplicate_members']==2

def test_source_exact_cast_sub_div_algebra():
    report=verify_export_algebra(EXPORT.read_text())
    assert report['input_clipping'] is False and report['actual_model_or_runtime_execution'] is False

@pytest.mark.parametrize('old,new',[
    ('features.to(torch.float64)','features.to(torch.float32)'),
    ('value=(features.to(torch.float64)-self.feature_mean)/self.feature_std','value=torch.clamp((features.to(torch.float64)-self.feature_mean)/self.feature_std,-5,5)'),
    ("helper.make_node('Div',['center','std'],['normalized'])","helper.make_node('Mul',['center','std'],['normalized'])")])
def test_changed_normalization_source_rejected(old,new):
    source=EXPORT.read_text();assert old in source
    with pytest.raises(ValueError):verify_export_algebra(source.replace(old,new))

def test_normalization_has_no_input_clip_or_second_float_cast():
    x,y,m,s=fixture(1);x[0,:3]=[1000,.3,-.2];m[:3]=[.1,.2,.3];s[:3]=[.05,.3,.7]
    expected=(x.astype(np.float64)-m.astype(np.float64))/s.astype(np.float64)
    actual=normalize64(x,m,s)
    assert actual.dtype==np.float64 and actual[0,0]>10000
    assert actual.tobytes()==expected.tobytes()
    assert actual[0,1]!=np.float64(np.float32(actual[0,1]))

@pytest.mark.parametrize('change',['dtype','width','std','nan'])
def test_normalization_schema_failures(change):
    x,y,m,s=fixture(1)
    if change=='dtype':x=x.astype(np.float64)
    if change=='width':x=x[:,:1322]
    if change=='std':s[0]=0
    if change=='nan':x[0,0]=np.nan
    with pytest.raises(ValueError):normalize64(x,m,s)

def test_fixed_query_selection_all_72_first24_each_phase():
    phases=np.r_[np.zeros(3,np.int8),np.repeat(np.arange(3,dtype=np.int8),[99,819,100])]
    controls=np.r_[np.arange(3),np.arange(251,1269)]
    q=fixed_queries(phases,controls,old_rows=3)
    assert q.tolist()==list(range(3,27))+list(range(102,126))+list(range(921,945))
    assert controls[q[[0,24,48]]].tolist()==[251,350,1169]

@pytest.mark.parametrize('change',['clock','count','phase'])
def test_query_selection_rejects_changed_scope(change):
    p=np.repeat(np.arange(3,dtype=np.int8),[99,819,100]);c=np.arange(251,1269)
    if change=='clock':c[1]+=1
    if change=='count':p=p[:-1];c=c[:-1]
    if change=='phase':p[0]=1
    with pytest.raises(ValueError):fixed_queries(p,c,old_rows=0)

def test_proximity_ties_phase_and_context_with_block_boundaries():
    x,y,m,s=fixture(5);p=np.array([0,1,1,0,1]);x[0,1000]=10;x[1,0]=1;x[2,0]=1;x[3,0]=2;y[1,2]=.5
    records={r['view']:r for r in proximity(x,y,p,[4],m,s,old_rows=4,block_size=2)}
    assert records['current_global']['candidate_row']==0
    assert records['full_global']['candidate_row']==1
    assert records['full_same_phase']['candidate_row']==1
    assert records['current_same_phase']['candidate_row']==1
    assert records['current_global']['normalized_prior_RMS']==np.sqrt(100/23)
    assert records['full_global']['target_difference_rad'][2]==-.5

def test_proximity_matches_independent_small_bruteforce():
    rng=np.random.default_rng(5);x=rng.standard_normal((11,1323)).astype(np.float32);y=rng.normal(size=(11,23))
    m=rng.normal(size=1323).astype(np.float32);s=rng.uniform(.1,2,size=1323).astype(np.float32);p=np.arange(11)%3
    records=proximity(x,y,p,[9,10],m,s,old_rows=9,block_size=3)
    z=(x.astype(np.float64)-m.astype(np.float64))/s.astype(np.float64)
    for row in records:
        q=row['query_row'];ids=np.arange(9)
        if row['view'].endswith('same_phase'):ids=ids[p[ids]==p[q]]
        width=1000 if row['view'].startswith('current') else 1323
        dist=np.sum((z[ids,:width]-z[q,:width])**2,axis=1)
        assert row['candidate_row']==int(ids[np.argmin(dist)])
        np.testing.assert_allclose(row['distance_squared'],dist.min(),rtol=1e-14)

def fake_gate(tmp_path):
    def subject(role,path=None):return dict(path=str(path or tmp_path/(role+'.json')),sha256=hashlib.sha256(role.encode()).hexdigest())
    subjects={r:subject(r) for r in gate.ROLES}
    shared=tmp_path/'shared'
    for role,name in [('old_shared_manifest','output_manifest.json'),('old_nominal_context','nominal_context.npy'),('old_physical_context','physical_context.npy'),('normalization','normalization.npz'),('old_context_alignment','context_alignment.json')]:subjects[role]=subject(role,shared/name)
    subjects['collection_report']=subject('collection_report',tmp_path/'collection/report.json')
    subjects['new_rows']=subject('new_rows',tmp_path/'collection/expert_rows.npz')
    physical={k:subject(k,tmp_path/(k+'.npy')) for k in gate.PHYSICAL_KEYS}
    owner={f:True for f in ('owner_verification_passed','accounting_passed','completion_passed','numerical_completion_passed','processes_absent','all_postrun_pins_exact')}
    owner.update(raw_exit_known=True,raw_python_exit_code=0,exit_code=0)
    roles={'old_fit_report':'fit_report','old_training_request':'training_request','old_training_manifest':'training_manifest','old_shared_manifest':'shared_manifest','normalization':'normalization','old_context_alignment':'context_alignment'}
    owner['direct_subject_sha256']={name:subjects[role]['sha256'] for role,name in roles.items()}
    reports={'old_fit_owner':owner,'old_fit_report':dict(completed=True,ordinary_final_step=81000,architecture=[1323,512,512,23]),
        'old_fit_audit':dict(evidence_audit_passed=True,export_qualified=True,input_sha256={subjects[r]['path']:subjects[r]['sha256'] for r in roles}),
        'old_training_request':dict(paths={name:subjects[role]['path'] for role,name in [('old_centers','centers'),('old_pico','pico'),('old_walk002','walk002'),('physical_manifest','physical_manifest'),('contract','contract')]}),
        'old_training_manifest':dict(training_request_sha256=subjects['old_training_request']['sha256'],source_sha256={'width512_promoted.py':subjects['export_source']['sha256']},input_sha256={v['path']:v['sha256'] for v in [*subjects.values(),*physical.values()]}),
        'old_shared_manifest':dict(files={Path(subjects[r]['path']).name:subjects[r]['sha256'] for r in ('old_nominal_context','old_physical_context','normalization','old_context_alignment')}),
        'old_context_alignment':dict(nominal_chronological_history_and_prior_pairs=9899,physical_actual_prior_exact=True,physical_history_once_shifted_exact=True),
        'physical_manifest':dict(complete=True,arrays={k:dict(path=Path(v['path']).name,sha256=v['sha256']) for k,v in physical.items()}),
        'collection_report':dict(passed=True,collection_completed=True,model_fitting_authorized=False,rows=1018,control_start=251,control_stop_exclusive=1269,fresh_student_state_queries=1,
            outputs={'expert_rows.npz':subjects['new_rows']['sha256'],'normalization.npz':subjects['normalization']['sha256']},request_sha256=subjects['collection_request']['sha256'],
            input_sha256={subjects[r]['path']:subjects[r]['sha256'] for r in ('collection_qualification','collection_source_review')},source_sha256={'collector.py':'a'*64}),
        'collection_request':dict(root_selected_collection=True,model_fitting_authorized=False,subjects={name:subjects[role] for role,name in [('collection_qualification','qualification'),('collection_source_review','source_review')]},source_sha256={'collector.py':'a'*64}),
        'collection_qualification':dict(root_authorized_extraction=True,model_fitting_authorized=False,control_start=251,control_stop_exclusive=1269,rows=1018),
        'collection_source_review':dict(source_review_pass=True,source_sha256={'collector.py':'a'*64})}
    return subjects,reports,physical

def test_minimal_collector_admission_no_extra_data_review(tmp_path):
    s,r,p=fake_gate(tmp_path);gate.report_gate(s,r,p)
    assert 'collection_review' not in gate.ROLES

@pytest.mark.parametrize('change',['owner','old_audit','export_source','request','prior','history','collector_complete','scope','normalization','qualification','source','output_path','physical','physical_missing'])
def test_gate_corruptions(tmp_path,change):
    s,r,p=fake_gate(tmp_path)
    if change=='owner':r['old_fit_owner']['processes_absent']=False
    if change=='old_audit':r['old_fit_audit']['input_sha256'].clear()
    if change=='export_source':r['old_training_manifest']['source_sha256']['width512_promoted.py']='b'*64
    if change=='request':r['old_training_manifest']['training_request_sha256']='b'*64
    if change=='prior':r['old_context_alignment']['physical_actual_prior_exact']=False
    if change=='history':r['old_context_alignment']['physical_history_once_shifted_exact']=False
    if change=='collector_complete':r['collection_report']['collection_completed']=False
    if change=='scope':r['collection_report']['control_start']=250
    if change=='normalization':r['collection_report']['outputs']['normalization.npz']='b'*64
    if change=='qualification':r['collection_qualification']['root_authorized_extraction']=False
    if change=='source':r['collection_request']['source_sha256']={'collector.py':'b'*64}
    if change=='output_path':s['new_rows']['path']=str(tmp_path/'wrong.npz')
    if change=='physical':p['advanced_history']['sha256']='b'*64
    if change=='physical_missing':p.pop('advanced_history')
    with pytest.raises(ValueError):gate.report_gate(s,r,p)

def test_path_alias_membership_is_literal_and_conflicts_fail():
    gate.member({'E:\\case\\file':'a'},dict(path='/mnt/e/case/file',sha256='a'))
    with pytest.raises(ValueError):gate.member({'E:/case/file':'a','/mnt/e/case/file':'b'},dict(path='E:/case/file',sha256='a'))
    with pytest.raises(ValueError):gate.member({'E:/other':'a'},dict(path='E:/case/file',sha256='a'))

def test_unselected_request_stops_before_hash_or_array_reads(tmp_path,monkeypatch):
    p=tmp_path/'unselected.json';p.write_text(json.dumps(dict(root_selected_saved_diagnosis=False)))
    monkeypatch.setattr(gate,'sha',lambda _:pytest.fail('Unselected request reached hashing'))
    with pytest.raises(ValueError,match='unselected'):gate.admit(p,p,'x',tmp_path)

def test_unused_object_metadata_rejected_before_load(tmp_path):
    p=tmp_path/'fake.npz';np.savez(p,features=np.zeros((1,2),np.float32),unused=np.array([object()],object))
    with pytest.raises(ValueError,match='object dtype'):load_numeric(p,{'features':((1,2),'float32')})

@pytest.mark.parametrize('dtype,shape',[('float64',(2,3)),('float32',(3,2))])
def test_numeric_header_schema_rejected(tmp_path,dtype,shape):
    p=tmp_path/'fake.npy';np.save(p,np.zeros(shape,dtype))
    with pytest.raises(ValueError,match='schema'):loader.checked_npy(p,(2,3),'float32')

def test_copied_label_cast_retains_original_float64_to_float32_order():
    y=np.full((2,23),.123456789,np.float64);default=np.linspace(-.2,.2,23);span=np.linspace(.1,2,23,dtype=np.float32)
    assert normalized_labels(y,default,span).tobytes()==((y-default)/span.astype(np.float64)).astype(np.float32).tobytes()

def synthetic_assembly(monkeypatch,corruption=None):
    # Full fixed counts, but broadcast-generated fake arrays; no task file opens.
    paths={role:role for role in gate.ROLES};physical={k:k for k in gate.PHYSICAL_KEYS}
    limits=np.tile(np.array([-1.,1.]),(23,1));span=np.full(23,2,np.float32);default=np.zeros(23)
    mean=np.zeros(1323,np.float32);std=np.ones(1323,np.float32)
    norm=dict(feature_mean=mean,feature_std=std,context_mean=mean[1000:],context_std=std[1000:],default_q=default,joint_span=span,joint_limits=limits)
    archives={'normalization':norm};arrays={}
    for role,n,codes in [('old_centers',3057,[0,1,2]),('old_pico',5980,[3]),('old_walk002',867,[4])]:
        cs=np.concatenate([np.arange(250,250+sum(loader.COUNTS[d]),dtype=np.int64) for d in codes])
        archives[role]=dict(features=np.broadcast_to(np.arange(1069,dtype=np.float32),(n,1069)),expert_target=np.zeros((n,23)),
            control=cs,source_frame=cs+11,previous_action=np.zeros((n,23),np.float32),history=np.zeros((n,300),np.float32),joint_span=span,joint_limits=limits,
            dataset=np.concatenate([np.full(sum(loader.COUNTS[d]),d,np.int64) for d in codes]),
            phase=np.concatenate([np.repeat(np.arange(3,dtype=np.int64),loader.COUNTS[d]) for d in codes]))
    arrays['old_nominal_context']=np.zeros((9904,323),np.float32)
    arrays['endpoint_features']=np.broadcast_to(np.arange(1069,dtype=np.float32),(3054,1069))
    arrays['label_fixed_map_target']=np.zeros((3054,23))
    arrays['old_physical_context']=np.zeros((3054,323),np.float32);arrays['old_physical_context'][:,:23]=.5
    arrays['policy_actual_normalized_action']=np.full((3054,23),.5,np.float32);arrays['advanced_history']=np.zeros((3054,300),np.float32)
    arrays['policy_applied_target']=np.full((3054,23),.25)
    arrays['dataset']=np.repeat(np.arange(3,dtype=np.int64),1018);arrays['successor_control']=np.tile(np.arange(251,1269,dtype=np.int64),3);arrays['start_control']=arrays['successor_control']-1
    for k in ('teacher_feedback_clipped','teacher_native_clipped'):arrays[k]=np.zeros((3054,23),bool)
    cf=np.zeros((1018,1323),np.float32);cf[:,0]=.3;cf[:,1000:1023]=.75;cf[:,1023:]=.125;cs=np.arange(251,1269,dtype=np.int64)
    new=dict(causal_features=cf,features=cf[:,:1000],context=cf[:,1000:],incoming_prior=cf[:,1000:1023],incoming_history=cf[:,1023:],expert_target=np.zeros((1018,23)),
        control=cs,source_frame=cs+11,phase=np.repeat(np.arange(3,dtype=np.int8),[99,819,100]),first_student_state_query=np.r_[True,np.zeros(1017,bool)],
        plan_control=251+5*((cs-251)//5),plan_local=(cs-251)%5,feedback_clipped=np.zeros((1018,23),bool),native_clipped=np.zeros((1018,23),bool))
    archives['new_rows']=new
    if corruption=='raw_prior':arrays['policy_actual_normalized_action']=np.zeros((3054,23),np.float32)
    if corruption=='history':new['incoming_history']=new['incoming_history'].copy();new['incoming_history'][1,0]+=1
    if corruption=='phase':new['phase'][0]=1
    if corruption=='target':new['expert_target'][0,0]=1.01
    if corruption=='normalization':norm['context_std']=np.full(323,2,np.float32)
    def fake_load(path,spec):
        rows=archives[str(path)]
        for k,(shape,dtype) in spec.items():assert rows[k].shape==shape and str(rows[k].dtype)==dtype
        return {k:rows[k] for k in spec},{}
    def fake_npy(path,shape,dtype):
        a=arrays[str(path)];assert a.shape==shape and str(a.dtype)==dtype
        return a
    monkeypatch.setattr(loader,'load_numeric',fake_load);monkeypatch.setattr(loader,'checked_npy',fake_npy)
    contract=dict(default_q=default.tolist(),joint_limits=limits.tolist(),kp=[2.]*23,training_effort=[4.]*23)
    return loader.assemble(paths,physical,lambda _:contract)

def test_full_fixed_count_synthetic_assembly(monkeypatch):
    x,y,labels,norm,metadata=synthetic_assembly(monkeypatch)
    assert x.shape==(13976,1323) and y.shape==(13976,23)
    assert np.bincount(metadata['origin']).tolist()==[9904,3054,1018]
    assert metadata['control'][OLD:].tolist()==list(range(251,1269))
    np.testing.assert_array_equal(x[0,:1000],np.arange(1069,dtype=np.float32)[RETAINED])
    assert np.all(x[9904:OLD,1000:1023]==.5) and np.all(x[OLD:,1000:1023]==.75)
    assert not metadata['clipping_flags_known'][:9904].any() and metadata['clipping_flags_known'][9904:].all()
    assert fixed_queries(metadata['phase'],metadata['control']).shape==(72,)

@pytest.mark.parametrize('corruption',['raw_prior','history','phase','target','normalization'])
def test_assembly_rejects_misaligned_context_or_labels(monkeypatch,corruption):
    with pytest.raises(ValueError):synthetic_assembly(monkeypatch,corruption)

def test_source_imports_exclude_task_models_and_native():
    directory=Path(__file__).resolve().parent
    for path in directory.glob('*.py'):
        if path.name.startswith('test_'):continue
        for node in ast.walk(ast.parse(path.read_text())):
            names=[a.name for a in node.names] if isinstance(node,ast.Import) else [node.module] if isinstance(node,ast.ImportFrom) else []
            assert not any(n and n.split('.')[0] in {'torch','onnx','onnxruntime','mujoco','mjbatch','ctypes','subprocess'} for n in names),path.name
