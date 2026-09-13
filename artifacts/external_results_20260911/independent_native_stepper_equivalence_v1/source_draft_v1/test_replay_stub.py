"""Deterministic scripted captures and fake APIs, zero real native/model calls."""
from types import SimpleNamespace
from pathlib import Path
import json
import numpy as np
import pytest
from clock_core import CapturedStep,WarningLedger
from capture_schema import pack,decode
from replay_core import Segment,Evidence,ParityFailure,replay_segment,validate_segment
from counted_api import CountedAPI,mjb_pair
from runner import require_review_subject,require_pinned_roles,sha

def scripted(count):
    captures=[];t=0.
    for i in range(count+1):
        q=np.zeros(30,np.float64);q[2]=.75;q[3]=1.;q[7:]=i*.00001
        dq=np.zeros(29,np.float64);dq[6:]=.005
        v=np.zeros(291,np.float64);v[0]=t;v[1:31]=q;v[31:60]=dq
        torque=np.full(23,i*.0002,np.float64);v[89:112]=torque
        captures.append(CapturedStep(t,pack(v,q,dq,torque),torque.tobytes(),WarningLedger((0,)*8,(0,)*8)))
        t+=.002
    return captures

def segment(captures,start,steps,name,expert=False,fail=False):
    assert start%10==0
    selected=captures[start:start+steps+1];n=(steps+9)//10
    integration=np.array([decode(c.state).integration for c in selected])
    q=np.array([decode(c.state).qpos for c in selected]);dq=np.array([decode(c.state).qvel for c in selected])
    boundaries=[0]+[min((i+1)*10,steps) for i in range(n)]
    z=dict(target=np.zeros((n,23),np.float64),global_control=np.arange(start//10,start//10+n,dtype=np.int64),
        physics_substeps=np.array([10]*(n-1)+[steps-10*(n-1)],np.int64),
        control_integration_before=integration[np.arange(n)*10].copy(),qpos=q[boundaries],qvel=dq[boundaries],
        physics_qpos=q,physics_qvel=dq,physics_torque=np.array([np.frombuffer(c.torque,np.float64) for c in selected[1:]]),
        physics_time=np.array([c.simulation_time for c in selected],np.float64),
        physics_expected_time=np.array([c.simulation_time for c in selected],np.float64),
        initial_integration=integration[0],final_integration=integration[-1],integration_state_spec=np.asarray(8191,np.int64),
        physics_warning_lastinfo=np.zeros((steps+1,8),np.int32))
    z['physics_actuator_force' if expert else 'physics_actuator_torque']=np.array([decode(c.state).actuator_force for c in selected[1:]])
    z['physics_warning_number' if expert else 'physics_warning_counts']=np.zeros((steps+1,8),np.int32)
    issue=(start//10+n-1,steps-10*(n-1),'native_joint_bound') if fail else None
    return Segment(name,z,start//10,n,n+100 if fail else n,steps,issue)

class FakeStepper:
    def __init__(self,captures,fail_at=None):
        self.script=captures;self.initial_capture=captures[0];self.last_capture=captures[0]
        self.expected_time=captures[0].simulation_time
        self.attempted=self.returned=self.captured=self.verified=self.capture_attempts=self.verification_attempts=0
        self.fail_at=fail_at;self.failure=None;self.restore_calls=1;self.boundary_calls=0
    def boundary_snapshot(self):
        assert self.failure is None;self.boundary_calls+=1
        return decode(self.last_capture.state).integration.tobytes(),{'dof_pos':np.zeros(23,np.float32)}
    def step(self,target):
        assert self.returned==self.verified and self.failure is None
        self.attempted+=1;self.returned+=1;self.expected_time+=.002
    def capture_step(self):
        self.capture_attempts+=1;self.captured+=1;self.last_capture=self.script[self.returned];return self.last_capture
    def verify_step(self,capture):
        self.verification_attempts+=1
        if self.returned==self.fail_at:self.failure=SimpleNamespace(reason='STRICT_NATIVE_VIOLATION');return 'native_joint_bound'
        self.verified+=1;return None
    def counters(self):return {k:getattr(self,k) for k in ('attempted','returned','captured','verified','capture_attempts','verification_attempts')}

def test_continuous_main_hold_no_reset(tmp_path):
    c=scripted(20);adapter=FakeStepper(c);proof=Evidence(tmp_path/'proof')
    replay_segment(adapter,segment(c,0,10,'main',expert=True),proof)
    replay_segment(adapter,segment(c,10,10,'hold',expert=True),proof)
    assert adapter.restore_calls==1 and adapter.returned==adapter.verified==20
    assert proof.segment_results[1]['requested_segment_complete'] is True
    proof.save();z=np.load(proof.output/'captured_trace.npz');assert z['packed_capture'].shape==(20,2984)

def test_exact_partial_strict_failure_no_extra_steps(tmp_path):
    c=scripted(18);adapter=FakeStepper(c,18);proof=Evidence(tmp_path/'proof')
    r=replay_segment(adapter,segment(c,0,18,'failure',fail=True),proof)
    assert r['actual_issue']==(1,8,'native_joint_bound')
    assert adapter.returned==adapter.captured==18 and adapter.verified==17
    assert not r['requested_segment_complete'] and len(proof.captures)==18

def test_missing_expected_failure_preserves_last_capture(tmp_path):
    c=scripted(18);adapter=FakeStepper(c);proof=Evidence(tmp_path/'proof')
    with pytest.raises(ParityFailure,match='not reproduced'):replay_segment(adapter,segment(c,0,18,'failure',fail=True),proof)
    assert len(proof.captures)==18 and adapter.verification_attempts==18

def test_unexpected_early_failure_stops(tmp_path):
    c=scripted(20);adapter=FakeStepper(c,3);proof=Evidence(tmp_path/'proof')
    with pytest.raises(ParityFailure,match='unexpected'):replay_segment(adapter,segment(c,0,20,'main'),proof)
    assert adapter.returned==3 and len(proof.captures)==3

def test_saved_force_mismatch_retains_exact_attempt(tmp_path):
    c=scripted(10);s=segment(c,0,10,'main');s.arrays['physics_actuator_torque'][0,0]+=1
    adapter=FakeStepper(c);proof=Evidence(tmp_path/'proof')
    with pytest.raises(ParityFailure,match='actual_force'):replay_segment(adapter,s,proof)
    assert len(proof.captures)==1 and adapter.verification_attempts==1
    z=np.load(proof.output/'first_mismatch.npz');assert z['substep']==1 and not np.array_equal(z['actual'],z['expected'])

def test_full291_hidden_field_mismatch_before_step(tmp_path):
    c=scripted(10);s=segment(c,0,10,'main');s.arrays['control_integration_before'][0,60]=123.
    s.arrays['initial_integration'][60]=123.
    adapter=FakeStepper(c);proof=Evidence(tmp_path/'proof')
    with pytest.raises(ParityFailure,match='full291'):replay_segment(adapter,s,proof)
    assert adapter.returned==0

@pytest.mark.parametrize('bad',['warning_dtype','source_index','step_total','time'])
def test_schema_rejects_before_replay(bad):
    s=segment(scripted(10),0,10,'main')
    if bad=='warning_dtype':s.arrays['physics_warning_counts']=s.arrays['physics_warning_counts'].astype(np.int64)
    if bad=='source_index':s.arrays['global_control'][0]=1
    if bad=='step_total':s.arrays['physics_substeps'][0]=9
    if bad=='time':s.arrays['physics_expected_time'][1]+=.01
    with pytest.raises(AssertionError):validate_segment(s)

class FakeAPI:
    __version__='3.2.3'
    def __init__(self,partial=False):self.partial=partial;self.steps=0;self.saves=0
    def __getattr__(self,k):
        if k.startswith('get_mjcb_'):return lambda:None
        raise AttributeError(k)
    def mj_version(self):return 323
    def mj_versionString(self):return '3.2.3'
    def mj_sizeModel(self,model):return 3
    def mj_saveModel(self,model,filename,buffer):
        self.saves+=1;buffer[:2 if self.partial else 3]=np.frombuffer(b'ABC'[:2 if self.partial else 3],np.uint8)
    def mj_step(self,*args):self.steps+=1

def test_two_fill_witness_and_budget(tmp_path):
    native=FakeAPI();api=CountedAPI(native,0,2);model=SimpleNamespace(nplugin=0,npluginstate=0)
    raw,r=mjb_pair(model,api,tmp_path,'witness');assert raw==b'ABC' and r['passed']
    assert api.serialization_attempted==api.serialization_returned==native.saves==2
    with pytest.raises(RuntimeError,match='budget'):api.mj_saveModel(None,None,None)
    with pytest.raises(RuntimeError,match='budget'):api.mj_step(None,None)
    assert native.saves==2 and native.steps==0

def test_incomplete_mjb_preserves_both_buffers(tmp_path):
    api=CountedAPI(FakeAPI(partial=True),0,2);model=SimpleNamespace(nplugin=0,npluginstate=0)
    with pytest.raises(AssertionError,match='different-fill'):mjb_pair(model,api,tmp_path,'witness')
    r=json.loads((tmp_path/'witness_serialization.json').read_text());assert not r['passed']
    assert len(r['returned_files'])==2 and (tmp_path/'witness_0.mjb').read_bytes()!=(tmp_path/'witness_1.mjb').read_bytes()

def test_every_native_serialization_buffer_retained_without_extra_call(tmp_path):
    native=FakeAPI();api=CountedAPI(native,0,2,tmp_path/'all_buffers');model=SimpleNamespace(nplugin=0,npluginstate=0)
    mjb_pair(model,api,tmp_path,'witness')
    assert native.saves==2 and len(api.serialization_records)==2
    assert all(r['native_returned'] for r in api.serialization_records)
    assert (tmp_path/'all_buffers/00.mjb').read_bytes()==(tmp_path/'all_buffers/01.mjb').read_bytes()==b'ABC'

def test_failed_native_serialization_preserves_partial_buffer(tmp_path):
    class Throws(FakeAPI):
        def mj_saveModel(self,model,filename,buffer):
            self.saves+=1;buffer[0]=65;raise RuntimeError('scripted native save fault')
    native=Throws();api=CountedAPI(native,0,2,tmp_path/'all_buffers');model=SimpleNamespace(nplugin=0,npluginstate=0)
    with pytest.raises(RuntimeError,match='save fault'):mjb_pair(model,api,tmp_path,'witness')
    assert native.saves==api.serialization_attempted==1 and api.serialization_returned==0
    assert (tmp_path/'all_buffers/00.mjb').read_bytes()==bytes([65,0xA5,0xA5])
    assert api.serialization_records[0]['native_returned'] is False

@pytest.mark.parametrize('budget',[-1,1.5,True])
def test_invalid_budget(budget):
    with pytest.raises(ValueError):CountedAPI(FakeAPI(),budget,2)

def test_wrong_review_subject_cannot_hide_in_other_metadata(tmp_path):
    request=tmp_path/'request.json';request.write_text('{}');digest=sha(request)
    good=dict(request_subject=dict(path=str(request),sha256=digest))
    require_review_subject(good,request,digest)
    with pytest.raises(AssertionError):require_review_subject(dict(request_subject=dict(path=str(tmp_path/'other'),sha256=digest),other=digest),request,digest)
    with pytest.raises(AssertionError):require_review_subject(dict(request_subject=dict(path=str(request),sha256='a'*64),other=digest),request,digest)
    with pytest.raises(KeyError):require_review_subject(dict(other=digest),request,digest)

def pinned_roles_fixture(tmp_path):
    bundle=tmp_path/'bundle';(bundle/'walk003').mkdir(parents=True);(bundle/'meshes').mkdir()
    names=('native_prepared.xml','prepared_model_arrays.npz','contract.json','walk003/native_original.npz','meshes/a.stl')
    for name in names:(bundle/name).write_bytes(name.encode())
    manifest=dict(portable_xml_sha256=sha(bundle/'native_prepared.xml'),prepared_arrays_sha256=sha(bundle/'prepared_model_arrays.npz'),
        cases={'walk003':{'native_original.npz':sha(bundle/'walk003/native_original.npz')}},meshes={'a.stl':sha(bundle/'meshes/a.stl')})
    (bundle/'manifest.json').write_text(json.dumps(manifest))
    r=dict(stage='replay',native_bundle=str(bundle),traces={})
    for key in ('canonical_fixture','mjb_witness_report','expected_model_mjb'):
        path=tmp_path/key;path.write_bytes(key.encode());r[key]=str(path)
    for key in ('expert_main','expert_hold','direct_failure'):
        path=tmp_path/key;path.write_bytes(key.encode());r['traces'][key]=str(path)
    pins={p.resolve():sha(p) for p in tmp_path.rglob('*') if p.is_file()}
    return r,pins

@pytest.mark.parametrize('role',['canonical_fixture','mjb_witness_report','expected_model_mjb','expert_main','expert_hold','direct_failure','mesh','xml'])
def test_unpinned_consumed_role_rejected(tmp_path,role):
    r,pins=pinned_roles_fixture(tmp_path);require_pinned_roles(r,pins)
    if role in r['traces']:path=r['traces'][role]
    elif role=='mesh':path=tmp_path/'bundle/meshes/a.stl'
    elif role=='xml':path=tmp_path/'bundle/native_prepared.xml'
    else:path=r[role]
    del pins[Path(path).resolve()]
    with pytest.raises(AssertionError,match='not pinned'):require_pinned_roles(r,pins)
