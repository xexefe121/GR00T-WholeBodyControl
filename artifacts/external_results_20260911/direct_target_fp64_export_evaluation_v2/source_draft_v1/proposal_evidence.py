"""Proposal diagnostics and schema for the direct absolute-target adapter."""
from pathlib import Path
import numpy as np


def exact(a,b):
    a,b=np.asarray(a),np.asarray(b)
    return a.shape==b.shape and a.dtype==b.dtype and a.tobytes()==b.tobytes()


def normalized_actual_target(target,contract):
    return ((np.asarray(target)-np.asarray(contract['default_q']))*np.asarray(contract['kp'])/
            (.25*np.asarray(contract['training_effort']))).astype(np.float32)


class RecordingSession:
    """Forward inputs and outputs without changing values, call count or order."""
    def __init__(self,inner,name,evidence):self.inner=inner;self.name=name;self.evidence=evidence
    def run(self,output_names,input_feed,*args,**kwargs):
        for key,value in input_feed.items():
            self.evidence.current[self.name+'_input_'+key]=np.asarray(value).copy()
        output=self.inner.run(output_names,input_feed,*args,**kwargs)
        for index,value in enumerate(output):
            self.evidence.current[self.name+'_output_'+str(index)]=np.asarray(value).copy()
        return output


class ProposalEvidence:
    def __init__(self,runtime,contract):
        self.runtime=runtime;self.contract=contract;self.current={};self.attempts=0
        self.last_raw_proposal=None;self.last_actual_action=None
        runtime.head=RecordingSession(runtime.head,'head',self)
        for name in ('backward','actor'):
            runtime.seed.sessions[name]=RecordingSession(runtime.seed.sessions[name],name,self)

    def begin(self,control,qpos,qvel,integration,expected_time,warnings,lastinfo):
        self.attempts+=1
        self.current=dict(global_control=np.asarray(control,np.int64),qpos=np.asarray(qpos).copy(),
            qvel=np.asarray(qvel).copy(),integration_before=np.asarray(integration).copy(),
            expected_time=np.asarray(expected_time),warning_counts=np.asarray(warnings).copy(),
            warning_lastinfo=np.asarray(lastinfo).copy(),
            previous_action_before=self.runtime.seed.previous_action.copy(),
            recorded_controls_before=np.asarray(self.runtime.seed.recorded_controls,np.int64))
        for key,value in self.runtime.seed.history.data.items():
            self.current['history_before_'+key]=value.copy()
        self.last_raw_proposal=None;self.last_actual_action=None

    def accepted_proposal(self,proposed):
        # Evidence reconstructs the output. Runtime independently owns its phase
        # action convention: actual-target inverse only in the learned phase.
        raw=proposed['base_target']+proposed['delta']
        actual=normalized_actual_target(proposed['target'],self.contract)
        limits=np.asarray(self.contract['joint_limits'])
        if not exact(np.clip(raw,limits[:,0],limits[:,1]),proposed['target']):
            raise ValueError('Recorded raw proposal does not reconstruct original target.')
        self.last_raw_proposal=np.asarray(raw).copy()
        self.last_actual_action=actual.copy()
        self.current['raw_proposal']=self.last_raw_proposal.copy()
        self.current['actual_normalized_action']=actual.copy()
        for key,value in proposed.items():
            self.current['proposal_'+key]=np.asarray(value).copy()

    def preserve(self,path,reason,qpos,qvel,integration,expected_time,warnings,lastinfo):
        content=dict(self.current)
        content.update({'runtime_'+k:np.asarray(v).copy() for k,v in self.runtime.context.items()})
        content.update({'count_'+k:np.asarray(v,np.int64) for k,v in self.runtime.counts.items()})
        content['uncommitted_proposal']=np.asarray(self.runtime.pending is not None)
        content['forbidden_inference_calls']=np.asarray(self.runtime.forbidden_calls,np.int64)
        content.update(reason=np.asarray(str(reason)),qpos_at_failure=np.asarray(qpos).copy(),
            qvel_at_failure=np.asarray(qvel).copy(),integration_at_failure=np.asarray(integration).copy(),
            expected_time_at_failure=np.asarray(expected_time),warning_counts_at_failure=np.asarray(warnings).copy(),
            warning_lastinfo_at_failure=np.asarray(lastinfo).copy(),
            previous_action_after=self.runtime.seed.previous_action.copy(),
            recorded_controls_after=np.asarray(self.runtime.seed.recorded_controls,np.int64))
        for key,value in self.runtime.seed.history.data.items():content['history_after_'+key]=value.copy()
        np.savez_compressed(Path(path),**content)


TRAILING_SHAPES=dict(qpos=(30,),qvel=(29,),target=(23,),state=(52,),history=(300,),
    previous_action=(23,),action=(23,),base_target=(23,),delta=(23,),features=(1000,),normalized_head=(23,),
    joint_error=(23,),root_error=(3,),raw_proposal=(23,),actual_normalized_action=(23,),
    physics_qpos=(30,),physics_qvel=(29,),physics_torque=(23,),physics_actuator_torque=(23,),
    physics_warning_counts=(8,),physics_warning_lastinfo=(8,),
    control_integration_before=(291,),control_history_before=(300,),control_previous_action_before=(23,))
EMPTY_DTYPES={key:np.float32 for key in ('state','history','previous_action','action','delta','features',
    'actual_normalized_action','normalized_head','control_history_before','control_previous_action_before')}
EMPTY_DTYPES.update({key:np.int64 for key in ('source_frame','global_control','controller_mode','physics_substeps')})
EMPTY_DTYPES.update({key:np.int32 for key in ('physics_warning_counts','physics_warning_lastinfo')})


def trace_arrays(trace):
    result={}
    for key,values in trace.items():
        value=np.asarray(values)
        if len(value)==0:
            value=value.astype(EMPTY_DTYPES.get(key,np.float64))
            if key in TRAILING_SHAPES:value=value.reshape((0,)+TRAILING_SHAPES[key])
        result[key]=value
    commands=len(result['target']);steps=len(result['physics_torque'])
    if len(result['qpos'])!=commands+1 or len(result['qvel'])!=commands+1:
        raise ValueError('Actual command-boundary trace length mismatch.')
    for key in ('physics_qpos','physics_qvel','physics_time','physics_expected_time','physics_warning_counts','physics_warning_lastinfo'):
        if len(result[key])!=steps+1:raise ValueError('Native sample trace length mismatch: '+key)
    for key in ('source_frame','global_control','controller_mode','state','history','previous_action','action',
                'base_target','delta','features','normalized_head','raw_proposal','actual_normalized_action','physics_substeps'):
        if len(result[key])!=commands:raise ValueError('Applied control trace length mismatch: '+key)
    attempted=len(result['control_integration_before'])
    if attempted not in (commands,commands+1):raise ValueError('Attempted precontrol trace count mismatch.')
    for key in ('control_history_before','control_previous_action_before'):
        if len(result[key])!=attempted:raise ValueError('Attempted observation count mismatch.')
    if int(np.sum(result['physics_substeps']))!=steps:raise ValueError('Actual native substep count mismatch.')
    return result
