"""Private native23 experiment: collapse identical unguided dynamics, retain 9-lane costs.

No shared module changes. Guided line search dispatches directly to frozen code.
The full mjSTATE_INTEGRATION buffer is broadcast along with bound inputs/outputs;
copying qpos/qvel alone would leave hidden state inconsistent for later rollouts.
"""
import numpy as np
import mujoco
from gear_sonic.utils.g1_true23_mjbatch_mpc import Native23Tracker


class SingleLaneNative23Tracker(Native23Tracker):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if (self.nq,self.nv,self.nu)!=(30,29,23) or self.model.na or self.model.nplugin:
            raise ValueError('private optimization supports only stateless native23 actuator model')
        self._single_unguided=False
        self._single_id=np.asarray([0],np.int64)
        self.single_lane_steps=0
        self.guided_advance_calls=0
        # Existing bindings return their same live buffers. Full integration state
        # also preserves act/history/equality/mocap/user/plugin components if present.
        self._state=self.line.bind('state')
        self._broadcast_fields=[self.line.bind(name) for name in (
            'qpos','qvel','ctrl','qacc_warmstart','time','warning',
            'qfrc_actuator','xpos','xquat')]
        self._external=[self.line.bind(name) for name in ('qfrc_applied','xfrc_applied')]
        self._override_slices={}
        for name in ('TIME','QPOS','QVEL','CTRL','WARMSTART'):
            flag=getattr(mujoco.mjtState,'mjSTATE_'+name)
            begin=mujoco.mj_stateSize(self.model,int(flag)-1)
            width=mujoco.mj_stateSize(self.model,flag)
            self._override_slices[name]=slice(begin,begin+width)

    def _broadcast(self):
        self._state[1:]=self._state[:1]
        for values in self._broadcast_fields:
            values[1:]=values[:1]

    def rollout(self,x0,us,gains=None):
        # External force interventions and heterogeneous existing soft-warning
        # histories retain the unmodified path. Native hard rollouts reset warnings.
        collapse=(gains is None and all(not np.any(v) for v in self._external)
                  and (self.feasibility is not None or
                       np.array_equal(self.warning,np.repeat(self.warning[:1],len(self.warning),axis=0))))
        self._single_unguided=collapse
        try:
            return super().rollout(x0,us,gains)
        finally:
            self._single_unguided=False

    def advance(self,x,u):
        # If all lanes were rejected before integration, frozen advance only
        # changes public input buffers. Its authoritative state stays untouched.
        if self.feasibility is not None and not self._rollout_valid.any():
            return super().advance(x,u)
        if self.feasibility is not None and self._rollout_knot==0:
            # Defer the rollout clock override until it is consumed by a step.
            self._state[:,self._override_slices['TIME']]=0.0
        # Dormant lanes' Batch input mirrors were not advanced by CopyOut. An
        # assignment equal to an old mirror (especially warm=0) can otherwise be
        # ignored, leaving the copied integration state with a nonzero warmstart.
        # Explicitly apply exactly the four overrides the frozen advance makes,
        # to the authoritative state as well as its usual public bound buffers.
        for name,value in (('QPOS',x[:,:self.nq]),('QVEL',x[:,self.nq:]),('CTRL',u),('WARMSTART',0.0)):
            self._state[:,self._override_slices[name]]=value
        if not self._single_unguided:
            self.guided_advance_calls+=1
            return super().advance(x,u)
        # Rollout generates identical x/u in every unguided lane. Preserve nine
        # cost/feasibility lanes while avoiding eight duplicate MuJoCo integrations.
        qpos,qvel,ctrl,warm=self.line_fields
        qpos[:],qvel[:],ctrl[:],warm[:]=x[:,:self.nq],x[:,self.nq:],u,0.0
        if self.feasibility is None:
            self.line.step(self._single_id,nstep=self.sub)
            self.single_lane_steps+=self.sub
            self._broadcast()
            return np.concatenate([qpos,qvel],axis=1)
        for substep in range(1,self.sub+1):
            if not self._rollout_valid.any():
                break
            if not self._rollout_valid.all():
                raise AssertionError('unguided feasibility lanes diverged before replication')
            self.line.step(self._single_id,nstep=1)
            self.single_lane_steps+=1
            self._broadcast()
            self._record_feasibility(qpos,qvel,force=self.line_force[:,6:],warning=self.warning[:,:,1],
                time=self.line_time,expected_time=(self._rollout_knot*self.sub+substep)*self.model.opt.timestep,
                substep=substep)
        if self._rollout_valid.any():
            self.line.forward(self._single_id)
            self._broadcast()
            self._record_feasibility(qpos,qvel,force=self.line_force[:,6:],warning=self.warning[:,:,1],
                time=self.line_time,expected_time=(self._rollout_knot+1)*self.sub*self.model.opt.timestep,
                substep=self.sub)
        return np.concatenate([qpos,qvel],axis=1)
