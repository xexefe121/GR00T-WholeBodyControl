"""Exact native step/serialization budgets; usable with synthetic APIs."""
import hashlib
import numpy as np
from model_identity import CALLBACKS

class CountedAPI:
    def __init__(self,api,step_budget,serialization_budget,capture_directory=None):
        if type(step_budget) is not int or step_budget<0 or type(serialization_budget) is not int or serialization_budget<0:
            raise ValueError('nonnegative integer budgets required')
        self.inner=api;self.step_budget=step_budget;self.serialization_budget=serialization_budget
        self.step_attempted=self.step_returned=self.serialization_attempted=self.serialization_returned=0
        self.denied_step_calls=self.denied_serialization_calls=0
        self.serialization_records=[]
        self.capture_directory=capture_directory
        if capture_directory is not None:
            from pathlib import Path
            self.capture_directory=Path(capture_directory);self.capture_directory.mkdir(exist_ok=False)
    def __getattr__(self,name):return getattr(self.inner,name)
    def mj_step(self,*args,**kwargs):
        if self.step_attempted>=self.step_budget:
            self.denied_step_calls+=1;raise RuntimeError('native step budget exhausted; no retry')
        self.step_attempted+=1;value=self.inner.mj_step(*args,**kwargs);self.step_returned+=1;return value
    def mj_saveModel(self,*args,**kwargs):
        if self.serialization_attempted>=self.serialization_budget:
            self.denied_serialization_calls+=1;raise RuntimeError('MJB serialization budget exhausted')
        index=self.serialization_attempted;self.serialization_attempted+=1
        record=dict(index=index,native_returned=False)
        try:
            value=self.inner.mj_saveModel(*args,**kwargs)
            self.serialization_returned+=1;record['native_returned']=True
            return value
        except BaseException as exc:
            record['error']=repr(exc);raise
        finally:
            if self.capture_directory is not None:
                # Retain every actual native output, including partially filled
                # failed buffers. This performs no additional serialization.
                try:
                    buffer=kwargs['buffer'];raw=buffer.tobytes()
                    path=self.capture_directory/(str(index).zfill(2)+'.mjb')
                    with path.open('xb') as f:f.write(raw)
                    record.update(file=path.name,sha256=hashlib.sha256(raw).hexdigest(),bytes=len(raw),
                        dtype=buffer.dtype.str,shape=list(buffer.shape))
                except BaseException as exc:
                    record['capture_error']=repr(exc);raise
                finally:self.serialization_records.append(record)
            else:self.serialization_records.append(record)
    def counters(self):
        return dict(step_budget=self.step_budget,step_attempted=self.step_attempted,step_returned=self.step_returned,
            serialization_budget=self.serialization_budget,serialization_attempted=self.serialization_attempted,
            serialization_returned=self.serialization_returned,denied_step_calls=self.denied_step_calls,
            denied_serialization_calls=self.denied_serialization_calls,
            serialization_records=[dict(x) for x in self.serialization_records])

def mjb_pair(model,api,output,stem,expected=None):
    """Independent witness/fallback exit, exactly two distinct fill buffers."""
    from pathlib import Path
    import json,sys
    output=Path(output);context=dict(stem=stem,passed=False,returned_files=[])
    try:
        assert sys.byteorder=='little' and api.__version__=='3.2.3' and api.mj_version()==323 and api.mj_versionString()=='3.2.3'
        assert int(model.nplugin)==int(model.npluginstate)==0
        for name in CALLBACKS:assert getattr(api,'get_mjcb_'+name)() is None,name
        size=api.mj_sizeModel(model);assert type(size) is int and 0<size<=256*1024*1024
        raw=[]
        for index,fill in enumerate((0xA5,0x5A)):
            buffer=np.full(size,fill,np.uint8)
            context['current_index']=index;context['current_returned']=False
            try:
                api.mj_saveModel(model,filename=None,buffer=buffer);context['current_returned']=True
            finally:
                path=output/(stem+'_'+str(index)+'.mjb');path.write_bytes(buffer.tobytes())
                context['current_buffer_file']=path.name
                context['current_buffer_sha256']=hashlib.sha256(buffer.tobytes()).hexdigest()
            raw.append(buffer.tobytes());context['returned_files'].append(path.name)
        assert raw[0]==raw[1],'different-fill MJB serializations differ'
        if expected is not None:assert raw[0]==expected,'complete MJB differs from independent witness'
        context.update(passed=True,mjb_sha256=hashlib.sha256(raw[0]).hexdigest(),bytes=len(raw[0]))
        return raw[0],context
    except Exception as exc:
        context['error']=repr(exc);raise
    finally:
        context['api_counters']=api.counters()
        (output/(stem+'_serialization.json')).write_text(json.dumps(context,indent=2)+'\n')
