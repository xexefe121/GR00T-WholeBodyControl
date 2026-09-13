"""Direct absolute target phase; model/native creation occurs only in from_native."""
import copy
import time
import numpy as np
from direct_features import DirectFeatures
from runtime_common import *


def exact(a,b):
    a,b=np.asarray(a),np.asarray(b)
    return a.shape==b.shape and a.dtype==b.dtype and a.tobytes()==b.tobytes()


def native_output(head,default,span,limits):
    """Preserve old rounded f32 span, promote before both output operations."""
    if head.dtype!=np.float32 or head.shape!=(23,) or not np.isfinite(head).all():
        raise ValueError('finite normalized_target float32[23] required')
    if default.dtype!=np.float64 or span.dtype!=np.float32 or limits.dtype!=np.float64:
        raise ValueError('exact default64 / existing span32 / native limits64 required')
    if default.shape!=(23,) or span.shape!=(23,) or limits.shape!=(23,2):
        raise ValueError('native23 output contract shape')
    if not np.isfinite(default).all() or not np.isfinite(span).all() or not np.isfinite(limits).all() or np.any(span<=0) or np.any(limits[:,0]>limits[:,1]):
        raise ValueError('finite ordered native output contract required')
    delta=span.astype(np.float64)*head.astype(np.float64)
    raw=default+delta
    target=np.clip(raw,limits[:,0],limits[:,1])
    if not np.isfinite(raw).all():raise ValueError('nonfinite direct native proposal')
    return raw,target,delta


def actual_previous_action(target,c):
    return ((target-np.asarray(c['default_q']))*np.asarray(c['kp'])/
            (.25*np.asarray(c['training_effort']))).astype(np.float32)


class Phases:
    def __init__(self,timeline):
        by_name={p['name']:p for p in timeline['phases']}
        self.learned_start=by_name['acquisition_ramp']['control_start']
        self.terminal_start=by_name['returned_standing']['control_start']
        self.source_start=by_name['source_motion']['control_start']
        self.source_stop=by_name['source_motion']['control_stop']
        self.total=timeline['total_requested_controls']
        self.frame_offset=timeline['prehistory_frames']
        if (self.learned_start,self.terminal_start,self.source_start,self.source_stop,self.total,self.frame_offset)!=(250,1269,350,1169,1569,11):
            raise ValueError('selected canonical walk003 timeline identity mismatch')
        if by_name['initial_standing']['control_stop']!=self.learned_start:
            raise ValueError('startup boundary gap')
        ordered=timeline['phases']
        if ordered[0]['control_start']!=0 or ordered[-1]['control_stop']!=self.total:
            raise ValueError('lifecycle boundary mismatch')
        for left,right in zip(ordered,ordered[1:]):
            if left['control_stop']!=right['control_start']:raise ValueError('phase clock gap')

    def mode(self,control):
        if type(control) is not int or control<0:raise ValueError('integer nonnegative control')
        return 0 if control<self.learned_start else (1 if control<self.terminal_start else 2)


class CountedSession:
    """Guard forbidden phase calls and preserve current call inputs/outputs."""
    def __init__(self,session,name,owner):self.inner,self.name,self.owner=session,name,owner
    def run(self,names,feed):
        o=self.owner;mode=o.current_mode
        allowed=(mode==1) if self.name=='head' else (mode in (0,2))
        if not allowed:
            o.forbidden_calls+=1
            raise ValueError('forbidden '+self.name+' inference in mode '+str(mode))
        key=str(mode)+'_'+self.name
        for name,value in feed.items():o.context[self.name+'_input_'+name]=np.asarray(value).copy()
        o.context[self.name+'_current_call_returned']=np.asarray(False)
        o.counts[key+'_attempted']+=1
        returned=self.inner.run(names,feed)
        o.counts[key+'_returned']+=1
        for index,value in enumerate(returned):o.context[self.name+'_output_'+str(index)]=np.asarray(value).copy()
        o.context[self.name+'_current_call_returned']=np.asarray(True)
        return returned


class DirectStudentRuntime:
    def __init__(self,seed,head,features,c,span,timeline,base_infer):
        self.seed,self.c,self.features=seed,c,features
        self.default=np.asarray(c['default_q']);self.limits=np.asarray(c['joint_limits'])
        self.span=np.asarray(span).copy()
        if self.default.dtype!=np.float64 or self.limits.dtype!=np.float64 or self.span.dtype!=np.float32:
            raise ValueError('frozen default64, limits64 and existing span32 required')
        if self.span.shape!=(23,) or not np.isfinite(self.span).all() or np.any(self.span<=0):
            raise ValueError('positive native23 span required')
        # Equality validates the existing artifact; runtime scale remains its f32 bytes.
        if not exact(self.span,(self.limits[:,1]-self.limits[:,0]).astype(np.float32)):
            raise ValueError('span differs from original rounded native span')
        self.phases=Phases(timeline);self.base_infer=base_infer
        self.current_mode=None;self.context={};self.pending=None;self.forbidden_calls=0
        self.counts={str(mode)+'_'+name+'_'+stage:0 for mode in range(3)
                     for name in ('backward','actor','head') for stage in ('attempted','returned')}
        self.head=CountedSession(head,'head',self)
        for name in ('backward','actor'):self.seed.sessions[name]=CountedSession(seed.sessions[name],name,self)

    @classmethod
    def from_native(cls,native,c,original,motion,original29,head_path,span,timeline):
        # Only a future fully bound evaluator calls this constructor.
        if sys.platform=='win32' or np.__version__!='1.26.4':
            raise ValueError('selected evaluator requires pinned WSL NumPy1.26.4')
        from gear_sonic.utils.g1_true23_mjbatch_bfm_seed import Native23BFMRolloutSeed
        from student_linear_runtime import infer_base
        seed=Native23BFMRolloutSeed(native,c,original,ONNX,dependency_directory=DEPS,threads=1)
        import onnxruntime as ort
        settings=ort.SessionOptions();settings.intra_op_num_threads=1;settings.inter_op_num_threads=1
        settings.execution_mode=ort.ExecutionMode.ORT_SEQUENTIAL
        head=ort.InferenceSession(str(head_path),sess_options=settings,providers=['CPUExecutionProvider'])
        if len(head.get_inputs())!=1 or head.get_inputs()[0].name!='features' or head.get_inputs()[0].shape[-1]!=1000:
            raise ValueError('direct features1000 ONNX input required')
        if len(head.get_outputs())!=1 or head.get_outputs()[0].name!='normalized_target':
            raise ValueError('normalized absolute target ONNX output required')
        result=cls(seed,head,DirectFeatures(motion,original29,c),c,span,timeline,infer_base)
        binaries=list((Path(ort.__file__).parent/'capi').glob('*pybind11_state*.so'))
        if len(binaries)!=1:raise ValueError('one actual WSL ORT binary required')
        result.ort_binary_sha256=sha(binaries[0])
        return result

    def propose(self,control,qpos,qvel):
        tick=time.perf_counter()
        if self.pending is not None or control!=self.seed.recorded_controls:
            raise ValueError('uncommitted proposal or control clock mismatch')
        self.current_mode=self.phases.mode(control);self.context={}
        before_counts=dict(self.counts)
        previous=self.seed.previous_action.copy()
        sensed,terms=self.seed._terms(qpos,qvel,previous)
        staged_history=copy.deepcopy(self.seed.history)
        history=staged_history.before_update(terms)
        self.context.update(previous_action=previous.copy(),state=sensed.copy(),history=history.copy())
        for key,value in staged_history.data.items():self.context['staged_history_'+key]=value.copy()
        frame=control+self.phases.frame_offset
        if self.current_mode==1:
            x=self.features(qpos,qvel,frame)
            self.context['features']=x.copy()
            output=self.head.run(None,{'features':x[None]})
            if len(output)!=1 or output[0].shape!=(1,23) or output[0].dtype!=np.float32:
                raise ValueError('direct head output schema')
            normalized=output[0][0].copy()
            raw,target,delta=native_output(normalized,self.default,self.span,self.limits)
            base=self.default.copy()
            outgoing=actual_previous_action(target,self.c)
        else:
            # The original helper retains raw*5, goal construction and base math.
            raw_action,base,sensed=self.base_infer(self.seed,qpos,qvel,previous,history,frame,self.current_mode==2)
            normalized=np.zeros(23,np.float32)
            delta=np.zeros(23,np.float32)
            x=np.zeros(1000,np.float32) if self.current_mode==2 else self.features(qpos,qvel,frame)
            raw=base+delta
            target=np.clip(raw,self.limits[:,0],self.limits[:,1])
            outgoing=(raw_action+delta*np.asarray(self.c['kp'])/(.25*np.asarray(self.c['training_effort']))).astype(np.float32)
        if not np.isfinite(target).all() or not np.isfinite(outgoing).all():raise ValueError('invalid output/history action')
        expected=('head',) if self.current_mode==1 else ('backward','actor')
        for key,value in self.counts.items():
            mode,name,stage=key.split('_')
            wanted=int(int(mode)==self.current_mode and name in expected)
            if value-before_counts[key]!=wanted:raise ValueError('exact phase inference accounting mismatch: '+key)
        proposed=dict(target=target,base_target=base,delta=delta,previous_action=previous,action=outgoing,
            state=sensed,history=history,features=x,normalized_head=normalized,
            inference_ms=(time.perf_counter()-tick)*1000)
        self.context.update(raw_proposal=raw.copy(),actual_normalized_action=actual_previous_action(target,self.c))
        for key,value in proposed.items():self.context['proposal_'+key]=np.asarray(value).copy()
        self.pending=(control,staged_history,outgoing.copy(),{k:np.asarray(v).copy() for k,v in proposed.items()})
        return proposed

    def commit(self,proposed):
        if self.pending is None:raise ValueError('no proposed command to commit')
        control,staged,outgoing,saved=self.pending
        if control!=self.seed.recorded_controls or set(proposed)!=set(saved):raise ValueError('commit identity mismatch')
        if any(not exact(proposed[k],saved[k]) for k in saved):raise ValueError('proposal changed before command commit')
        for key,value in staged.data.items():self.seed.history.data[key][:]=value
        self.seed.previous_action=outgoing.copy();self.seed.recorded_controls+=1
        self.pending=None
        self.current_mode=None
