"""Native MuJoCo3.2.3 physics behind the existing dynamics-training interface.

mjbatch preserves double-precision integration state and solver warm starts.
CUDA tensors are observation mirrors. They are written back only for explicit
training resets, so reading float32 observations never projects physical state.
"""
from types import SimpleNamespace
import numpy as np
import torch
import mujoco
import mjbatch

class NativeBatchSimulation:
    def __init__(self,count,model,device,velocity,num_threads=8):
        if mujoco.__version__!='3.2.3':raise ValueError('native training requires MuJoCo3.2.3')
        self.model=model;self.count=count;self.device=device
        self.batch=mjbatch.Batch(model,count,num_threads=num_threads,forward=False)
        self.native={k:self.batch.bind(k) for k in ('qpos','qvel','ctrl','xpos','xmat','time','warning')}
        self.data=SimpleNamespace(**{k:torch.zeros(value.shape,device=device,dtype=torch.float64 if k=='time' else torch.float32)
            for k,value in self.native.items() if k!='warning'})
        self.data.physical_bad=torch.zeros(count,device=device,dtype=torch.bool)
        self.limits=model.jnt_range[1:].copy();self.velocity=np.asarray(velocity)
        self.pending=np.zeros(count,dtype=bool)
        self.batch.forward();self._read(derived=True)
        self.state_spec=int(mujoco.mjtState.mjSTATE_INTEGRATION)
        self.state_scratch=mujoco.MjData(model)
        self.state_fields={}
        for key in ('time','qpos','qvel','act','qacc_warmstart','ctrl','qfrc_applied','xfrc_applied',
                    'eq_active','mocap_pos','mocap_quat','userdata','plugin_state'):
            value=getattr(self.state_scratch,key)
            if np.asarray(value).size:self.state_fields[key]=self.batch.bind(key)

    def export_integration(self,ids):
        """Read full native integration state without a float32 round trip."""
        ids=np.asarray(ids,dtype=int)
        result=np.empty((len(ids),mujoco.mj_stateSize(self.model,self.state_spec)))
        for row,index in enumerate(ids):
            for key,values in self.state_fields.items():
                if key=='time':self.state_scratch.time=float(np.asarray(values[index]).reshape(-1)[0])
                else:getattr(self.state_scratch,key)[:]=values[index]
            mujoco.mj_getState(self.model,self.state_scratch,result[row],self.state_spec)
        return result,self.native['warning'][ids].copy()

    def import_integration(self,ids,states,warnings=None):
        """Explicit training reset only. Refresh mirrors after native restore."""
        ids=np.asarray(ids,dtype=int);states=np.asarray(states,dtype=np.float64)
        expected=(len(ids),mujoco.mj_stateSize(self.model,self.state_spec))
        if states.shape!=expected or not np.isfinite(states).all():raise ValueError('Invalid native integration reset')
        for row,index in enumerate(ids):
            mujoco.mj_setState(self.model,self.state_scratch,states[row],self.state_spec)
            for key,values in self.state_fields.items():values[index]=getattr(self.state_scratch,key)
        self.native['warning'][ids]=0 if warnings is None else warnings
        self.pending[ids]=False  # Never overwrite the native restore from observation mirrors.
        # mjbatch.forward refreshes every world, including those not reset.
        # Preserve their solver memory too: another world's episode ending
        # must not change this world's next physical integration step.
        warm=self.state_fields['qacc_warmstart'].copy()
        self.batch.forward()
        self.state_fields['qacc_warmstart'][:]=warm
        self._read(derived=True)

    def _read(self,derived=False):
        for key in ('qpos','qvel','time',*(['xpos','xmat'] if derived else [])):
            getattr(self.data,key).copy_(torch.from_numpy(self.native[key]))
        q,v=self.native['qpos'],self.native['qvel']
        bad=(~np.isfinite(q).all(-1))|(~np.isfinite(v).all(-1))
        bad|=((q[:,7:]<self.limits[:,0]-1e-6)|(q[:,7:]>self.limits[:,1]+1e-6)).any(-1)
        bad|=(np.abs(v[:,6:])>self.velocity).any(-1)
        bad|=(self.native['warning'][:,:,1]>0).any(-1)
        bad|=(q[:,2]<.25)|((1-2*np.sum(q[:,4:6]**2,axis=-1))<np.cos(1.2))
        self.data.physical_bad.copy_(torch.from_numpy(bad))

    def _write_resets(self):
        ids=np.flatnonzero(self.pending)
        if len(ids):
            gpu_ids=torch.as_tensor(ids,device=self.device)
            for key in ('qpos','qvel'):
                self.native[key][ids]=getattr(self.data,key)[gpu_ids].detach().cpu().numpy()
            self.pending[ids]=False

    def reset(self,ids):
        ids=np.ascontiguousarray(ids.detach().cpu().numpy(),dtype=np.int32)
        if not len(ids):return
        self.batch.reset(ids);self._read(derived=True);self.pending[ids]=True

    def forward(self):
        self._write_resets()
        # Scoring/FK refresh is not another solver-memory advancement.
        warm=self.state_fields['qacc_warmstart'].copy()
        self.batch.forward()
        self.state_fields['qacc_warmstart'][:]=warm
        self._read(derived=True)

    def step(self):
        self._write_resets()
        self.native['ctrl'][:]=self.data.ctrl.detach().cpu().numpy()
        self.batch.step();self._read()
