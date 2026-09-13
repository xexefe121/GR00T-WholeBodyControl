"""Complete pinned-version MJB identity; no model construction or runtime loading.

MJB covers mjModel, not arbitrary process-global native modifications. A later
runner must bind the actual library/bindings and own the model without concurrent
mutation. Python callback getters cannot detect direct external C pointer writes.
"""
import hashlib
import re
import sys
import numpy as np

CALLBACKS=('passive','control','contactfilter','sensor','time','act_dyn','act_gain','act_bias')


class FullModelIdentity:
    def __init__(self,model,api,expected_mjb,expected_sha256):
        if type(expected_mjb) is not bytes or not 0<len(expected_mjb)<=256*1024*1024:
            raise ValueError('bounded immutable complete expected MJB bytes required')
        if re.fullmatch('[0-9a-f]{64}',str(expected_sha256)) is None or hashlib.sha256(expected_mjb).hexdigest()!=expected_sha256:
            raise ValueError('expected complete MJB SHA256 differs')
        self.model,self.api=model,api
        self.expected=expected_mjb;self.sha256=expected_sha256
        self.serialization_attempted=self.serialization_returned=0
        self.checks=[];self.exit_verified=False
        self.verify('entry')

    def verify(self,phase):
        if phase=='exit':self.exit_verified=False
        record=dict(phase=str(phase),passed=False,serialization_attempted_before=self.serialization_attempted,
                    serialization_returned_before=self.serialization_returned)
        try:
            if sys.byteorder!='little' or self.api.__version__!='3.2.3' or self.api.mj_version()!=323 or self.api.mj_versionString()!='3.2.3':
                raise ValueError('exact little-endian MuJoCo3.2.3 API required')
            if int(self.model.nplugin)!=0 or int(self.model.npluginstate)!=0:
                raise ValueError('plugins are outside the qualified native model contract')
            for name in CALLBACKS:
                if getattr(self.api,'get_mjcb_'+name)() is not None:
                    raise ValueError('non-null exposed callback: '+name)
            size=self.api.mj_sizeModel(self.model)
            if type(size) is not int or size!=len(self.expected):raise ValueError('complete MJB size differs')
            # Different fills expose any unwritten output byte without parsing or
            # filtering the MJB. These are serializations, never physics calls.
            returned=[]
            for fill in (0xA5,0x5A):
                buffer=np.full(size,fill,np.uint8)
                self.serialization_attempted+=1
                self.api.mj_saveModel(self.model,filename=None,buffer=buffer)
                self.serialization_returned+=1
                returned.append(buffer.tobytes())
            record['actual_sha256']=hashlib.sha256(returned[0]).hexdigest()
            if returned[0]!=returned[1]:raise ValueError('MJB serialization incomplete or changed during identity capture')
            if returned[0]!=self.expected:raise ValueError('complete MJB identity changed')
            record['passed']=True
            if phase=='exit':self.exit_verified=True
        except Exception as exc:
            record['error']=str(exc)
            raise
        finally:
            record.update(serialization_attempted_after=self.serialization_attempted,serialization_returned_after=self.serialization_returned)
            self.checks.append(record)
        return dict(record)

    def summary(self):
        return dict(expected_sha256=self.sha256,entry_verified=bool(self.checks and self.checks[0]['passed']),
                    exit_verified=self.exit_verified,checks=[dict(v) for v in self.checks],
                    serialization_attempted=self.serialization_attempted,serialization_returned=self.serialization_returned,
                    process_global_identity_proven=False,portable_cross_platform_identity=False)
