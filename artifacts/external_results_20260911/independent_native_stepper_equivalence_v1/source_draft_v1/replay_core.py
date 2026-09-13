"""Replay comparison bookkeeping, independent of native runtime construction."""
from dataclasses import dataclass
import hashlib,json
from pathlib import Path
import numpy as np
from capture_schema import exact,decode


class ParityFailure(RuntimeError):pass

@dataclass(frozen=True)
class Segment:
    name:str
    arrays:dict
    global_start:int
    recorded_controls:int
    requested_controls:int
    recorded_steps:int
    expected_issue:tuple|None=None  # (global control, one-based substep, issue)

def archive(path):
    with np.load(path,allow_pickle=False) as z:
        arrays={k:z[k].copy() for k in z.files}
    for a in arrays.values():a.setflags(write=False)
    return arrays

def validate_segment(segment):
    z=segment.arrays;n=segment.recorded_controls;s=segment.recorded_steps
    aliases={
        'force':'physics_actuator_force' if 'physics_actuator_force' in z else 'physics_actuator_torque',
        'warnings':'physics_warning_number' if 'physics_warning_number' in z else 'physics_warning_counts'}
    shapes={'target':(n,23),'global_control':(n,),'physics_substeps':(n,),
        'control_integration_before':(n,291),'qpos':(n+1,30),'qvel':(n+1,29),
        'physics_qpos':(s+1,30),'physics_qvel':(s+1,29),'physics_torque':(s,23),
        aliases['force']:(s,23),'physics_time':(s+1,),'physics_expected_time':(s+1,),
        aliases['warnings']:(s+1,8),'physics_warning_lastinfo':(s+1,8),
        'initial_integration':(291,),'final_integration':(291,)}
    for key,shape in shapes.items():
        assert key in z and z[key].shape==shape,(segment.name,key)
        dtype=np.int64 if key in ('global_control','physics_substeps') else np.int32 if key in (aliases['warnings'],'physics_warning_lastinfo') else np.float64
        assert z[key].dtype==dtype,(segment.name,key,'dtype')
        assert np.isfinite(z[key]).all(),(segment.name,key,'finite')
    assert int(z['integration_state_spec'])==8191
    assert exact(z['global_control'],np.arange(segment.global_start,segment.global_start+n,dtype=np.int64))
    assert np.all((z['physics_substeps']>=1)&(z['physics_substeps']<=10)) and int(z['physics_substeps'].sum())==s
    assert np.all(z['physics_substeps'][:-1]==10)
    if segment.expected_issue is None:assert np.all(z['physics_substeps']==10) and n==segment.requested_controls
    else:
        c,sub,issue=segment.expected_issue
        assert c==segment.global_start+n-1 and sub==int(z['physics_substeps'][-1]) and issue=='native_joint_bound'
    assert exact(z['initial_integration'],z['control_integration_before'][0])
    assert exact(z['initial_integration'][1:31],z['physics_qpos'][0])
    assert exact(z['initial_integration'][31:60],z['physics_qvel'][0])
    assert exact(z['final_integration'][1:31],z['physics_qpos'][-1])
    assert exact(z['final_integration'][31:60],z['physics_qvel'][-1])
    assert exact(z['physics_time'],z['physics_expected_time'])
    return aliases

class Evidence:
    def __init__(self,output):
        self.output=Path(output);self.output.mkdir(exist_ok=False)
        self.captures=[];self.commands=[];self.times=[];self.warnings=[];self.lastinfo=[]
        self.controls=[];self.substeps=[];self.segments=[];self.boundaries=[]
        self.comparisons=0;self.first_mismatch=None;self.current={};self.segment_results=[]
    def compare(self,name,actual,expected):
        a,b=np.asarray(actual),np.asarray(expected)
        if not exact(a,b):
            if self.first_mismatch is None:
                self.first_mismatch=dict(name=name,**self.current,actual_shape=list(a.shape),expected_shape=list(b.shape),
                    actual_dtype=str(a.dtype),expected_dtype=str(b.dtype))
                np.savez_compressed(self.output/'first_mismatch.npz',actual=a,expected=b,**{
                    k:np.asarray(v) for k,v in self.current.items()})
            raise ParityFailure(name)
        self.comparisons+=1
    def append(self,capture):
        self.captures.append(bytes(capture.state));self.commands.append(bytes(capture.torque))
        self.times.append(capture.simulation_time);self.warnings.append(tuple(capture.warnings.counts))
        self.lastinfo.append(tuple(capture.warnings.lastinfo));self.controls.append(self.current['control'])
        self.substeps.append(self.current['substep']);self.segments.append(self.current['segment'])
    def snapshot(self,name,raw):
        self.boundaries.append((str(name),bytes(raw)))
    def save(self,name='captured_trace.npz'):
        n=len(self.captures)
        np.savez_compressed(self.output/name,
            packed_capture=np.frombuffer(b''.join(self.captures),np.uint8).reshape(n,2984),
            commanded_torque=np.frombuffer(b''.join(self.commands),np.float64).reshape(n,23),
            simulation_time=np.asarray(self.times,np.float64),warning_counts=np.asarray(self.warnings,np.int32).reshape(n,8),
            warning_lastinfo=np.asarray(self.lastinfo,np.int32).reshape(n,8),control=np.asarray(self.controls,np.int64),
            substep=np.asarray(self.substeps,np.int64),segment=np.asarray(self.segments,dtype='U32'),
            boundary_name=np.asarray([p[0] for p in self.boundaries],dtype='U96'),
            boundary_integration=np.frombuffer(b''.join(p[1] for p in self.boundaries),np.float64).reshape(len(self.boundaries),291))
    def fault(self,adapter):
        fault=adapter.failure
        if fault is None:return None
        (self.output/'adapter_fault_evidence.bin').write_bytes(fault.evidence)
        result=dict(reason=fault.reason,stage=fault.stage,attempted=fault.attempted,returned=fault.returned,
                    captured=fault.captured,detail=fault.detail,evidence_sha256=hashlib.sha256(fault.evidence).hexdigest())
        if fault.capture_return_evidence is not None:
            (self.output/'adapter_capture_return_evidence.bin').write_bytes(fault.capture_return_evidence)
            result['capture_return_evidence_sha256']=hashlib.sha256(fault.capture_return_evidence).hexdigest()
        return result

def compare_initial(adapter,segment,proof):
    z=segment.arrays;aliases=validate_segment(segment)
    raw,terms=adapter.boundary_snapshot()
    assert 'actions' not in terms
    proof.current=dict(segment=segment.name,control=segment.global_start,substep=0)
    proof.compare('segment_initial_full291',np.frombuffer(raw,np.float64),z['initial_integration'])
    proof.snapshot(segment.name+'_initial',raw)
    capture=adapter.initial_capture if adapter.returned==0 else adapter.last_capture
    v=decode(capture.state)
    for key,a,b in [('initial_qpos',v.qpos,z['physics_qpos'][0]),('initial_qvel',v.qvel,z['physics_qvel'][0]),
        ('initial_time',np.asarray(capture.simulation_time,np.float64),z['physics_time'][0]),
        ('initial_warning_counts',np.asarray(capture.warnings.counts,np.int32),z[aliases['warnings']][0]),
        ('initial_warning_lastinfo',np.asarray(capture.warnings.lastinfo,np.int32),z['physics_warning_lastinfo'][0])]:proof.compare(key,a,b)
    return aliases

def replay_segment(adapter,segment,proof):
    """No resets. Expert hold enters this same adapter after the main segment."""
    aliases=compare_initial(adapter,segment,proof);z=segment.arrays;s=0;issue_found=None
    before=adapter.returned
    for row,control in enumerate(z['global_control']):
        proof.current=dict(segment=segment.name,control=int(control),substep=0)
        raw,terms=adapter.boundary_snapshot();assert 'actions' not in terms
        proof.compare('precontrol_full291',np.frombuffer(raw,np.float64),z['control_integration_before'][row])
        proof.snapshot(segment.name+'_precontrol_'+str(int(control)),raw)
        for sub in range(1,int(z['physics_substeps'][row])+1):
            proof.current=dict(segment=segment.name,control=int(control),substep=sub)
            adapter.step(z['target'][row].copy())
            capture=adapter.capture_step();proof.append(capture)
            # Classify every returned capture through the new adapter exactly once,
            # then compare saved values. A saved mismatch cannot skip the oracle.
            issue=adapter.verify_step(capture);s+=1;v=decode(capture.state)
            for key,a,b in [('qpos',v.qpos,z['physics_qpos'][s]),('qvel',v.qvel,z['physics_qvel'][s]),
                ('command',np.frombuffer(capture.torque,np.float64),z['physics_torque'][s-1]),
                ('actual_force',v.actuator_force,z[aliases['force']][s-1]),
                ('time',np.asarray(capture.simulation_time,np.float64),z['physics_time'][s]),
                ('warning_counts',np.asarray(capture.warnings.counts,np.int32),z[aliases['warnings']][s]),
                ('warning_lastinfo',np.asarray(capture.warnings.lastinfo,np.int32),z['physics_warning_lastinfo'][s])]:proof.compare(key,a,b)
            proof.compare('independent_expected_time',np.asarray(adapter.expected_time,np.float64),z['physics_expected_time'][s])
            if issue is not None:
                issue_found=(int(control),sub,issue)
                if issue_found!=segment.expected_issue or s!=segment.recorded_steps:
                    raise ParityFailure('unexpected strict native rejection '+str(issue_found))
            elif segment.expected_issue is not None and s==segment.recorded_steps:
                raise ParityFailure('saved strict failure was not reproduced')
        proof.compare('control_qpos',v.qpos,z['qpos'][row+1]);proof.compare('control_qvel',v.qvel,z['qvel'][row+1])
    assert adapter.returned-before==s==segment.recorded_steps
    proof.compare('segment_final_full291',decode(adapter.last_capture.state).integration,z['final_integration'])
    proof.snapshot(segment.name+'_final',decode(adapter.last_capture.state).integration.tobytes())
    assert issue_found==segment.expected_issue
    result=dict(segment=segment.name,requested_controls=segment.requested_controls,recorded_controls=segment.recorded_controls,
        reproduced_steps=s,expected_issue=segment.expected_issue,actual_issue=issue_found,all_saved_samples_byte_exact=True,
        requested_segment_complete=segment.expected_issue is None,adapter_counters=adapter.counters())
    proof.segment_results.append(result)
    return result
