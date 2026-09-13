"""Bounded native-PD joint-limit preview from the received measured state.

Simulation candidate, not a recursive safety guarantee. No reference samples,
plant solver memory, clip identity, or prepared motion enter this calculation.
Every proposal uses the original native gains, target bounds and effort caps.
"""
import copy
import ctypes as ct
import time
import mujoco
import numpy as np

DOUBLE=ct.POINTER(ct.c_double)


class NativePreviewBackend:
    def __init__(self,library,guard):
        self.model=guard.model;self.lib=ct.CDLL(str(library))
        self.parallelism=self.lib.preview_parallelism() if hasattr(self.lib,'preview_parallelism') else 1
        self.lib.preview_create.argtypes=[ct.c_void_p]+[DOUBLE]*4+[ct.c_int,ct.c_int,ct.c_double]
        self.lib.preview_create.restype=ct.c_void_p
        self.lib.preview_destroy.argtypes=[ct.c_void_p]
        self.lib.preview_reset.argtypes=[ct.c_void_p]+[DOUBLE]*3
        self.lib.preview_limits.argtypes=[ct.c_void_p]+[DOUBLE]*3
        self.lib.preview_limits.restype=ct.c_int
        arrays=[np.ascontiguousarray(x,dtype=float) for x in (guard.kp,guard.kd,guard.effort,guard.limits)]
        self.address=self.lib.preview_create(self.model._address,*[x.ctypes.data_as(DOUBLE) for x in arrays],
            guard.steps,guard.delay,guard.reserve)
        if not self.address:raise ValueError('Native preview rejected its model or settings')
        if hasattr(self.lib,'preview_set_parallelism'):
            self.lib.preview_set_parallelism.argtypes=[ct.c_void_p,ct.c_int]
            self.parallelism=self.lib.preview_set_parallelism(self.address,guard.native_parallelism)
            if self.parallelism!=guard.native_parallelism:raise ValueError('Native preview thread setting rejected')
        elif guard.native_parallelism!=self.parallelism:
            raise ValueError('Preview library does not support requested thread setting')

    def reset(self,qpos,qvel,previous):
        q,v=[np.ascontiguousarray(x,dtype=float) for x in (qpos,qvel)]
        previous=None if previous is None else np.ascontiguousarray(previous,dtype=float)
        self.lib.preview_reset(self.address,q.ctypes.data_as(DOUBLE),v.ctypes.data_as(DOUBLE),
            None if previous is None else previous.ctypes.data_as(DOUBLE))

    def preview(self,target):
        target=np.ascontiguousarray(target,dtype=float);low=np.empty(23);high=np.empty(23)
        result=self.lib.preview_limits(self.address,target.ctypes.data_as(DOUBLE),
            low.ctypes.data_as(DOUBLE),high.ctypes.data_as(DOUBLE))
        if result<0:raise ValueError('Nonfinite native preview')
        return low,high

    def __del__(self):
        if getattr(self,'address',None):
            self.lib.preview_destroy(self.address);self.address=None


class NativePreviewGuard:
    def __init__(self, model, contract, *, steps=10, iterations=3, reserve_rad=1e-4,delay_substeps=0,library=None,native_parallelism=2):
        if (model.nq, model.nv, model.nu)!=(30,29,23) or abs(model.opt.timestep-.002)>1e-12:
            raise ValueError('Native preview requires the native23 2ms plant')
        if not 1<=steps<=20 or not 1<=iterations<=3 or not 0<reserve_rad<=.001 or not 0<=delay_substeps<=6:
            raise ValueError('Invalid bounded native preview settings')
        self.model=model;self.seed=mujoco.MjData(model)
        self.kp=np.asarray(contract['kp'],float);self.kd=np.asarray(contract['kd'],float)
        self.effort=np.asarray(contract['native_effort'],float)
        self.limits=np.asarray(contract['joint_limits'],float)
        self.steps=steps;self.iterations=iterations;self.reserve=reserve_rad
        if native_parallelism not in (1,2):raise ValueError('Preview threads must be 1 or 2')
        self.native_parallelism=native_parallelism
        self.delay=delay_substeps;self.previous=None
        self.native=NativePreviewBackend(library,self) if library is not None else None

    def _preview(self, target):
        if self.native is not None:return self.native.preview(target)
        low=np.full(23,-np.inf);high=low.copy()
        schedules=(0,self.delay) if self.delay and self.previous is not None else (0,)
        for delay in schedules:
            data=copy.copy(self.seed)
            for step in range(self.steps+self.delay):
                applied=self.previous if step<delay else target
                data.ctrl[:]=np.clip(self.kp*(applied-data.qpos[7:])-self.kd*data.qvel[6:],-self.effort,self.effort)
                mujoco.mj_step(self.model,data)
                if not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all():
                    raise ValueError('Nonfinite native preview')
                low=np.maximum(low,self.limits[:,0]+self.reserve-data.qpos[7:])
                high=np.maximum(high,data.qpos[7:]-self.limits[:,1]+self.reserve)
        return low,high

    def apply(self, qpos, qvel, requested,previous=None):
        started=time.perf_counter()
        qpos,qvel,requested=map(lambda x:np.asarray(x,dtype=float),(qpos,qvel,requested))
        if qpos.shape!=(30,) or qvel.shape!=(29,) or requested.shape!=(23,) or not all(
                np.isfinite(x).all() for x in (qpos,qvel,requested)):
            raise ValueError('Invalid native preview input')
        self.previous=None if previous is None else np.asarray(previous,dtype=float).copy()
        if self.previous is not None and (self.previous.shape!=(23,) or not np.isfinite(self.previous).all()):
            raise ValueError('Invalid previously applied native target')
        if self.native is not None:self.native.reset(qpos,qvel,self.previous)
        else:
            mujoco.mj_resetData(self.model,self.seed)
            self.seed.qpos[:]=qpos;self.seed.qvel[:]=qvel
            mujoco.mj_forward(self.model,self.seed)
        candidate=np.clip(requested,self.limits[:,0],self.limits[:,1])
        low,high=self._preview(candidate);calls=1
        best=candidate.copy();best_error=max(float(low.max()),float(high.max()))
        original_error=best_error
        for _ in range(self.iterations):
            error=np.maximum(low,high);active=error>0
            if not active.any():break
            direction=np.where(high>=low,1.,-1.)
            probe=candidate.copy()
            probe[active]-=direction[active]*.02
            probe=np.clip(probe,self.limits[:,0],self.limits[:,1])
            distance=np.abs(probe-candidate)
            probe_low,probe_high=self._preview(probe);calls+=1
            same_boundary=np.where(direction>0,probe_high,probe_low)
            response=(error-same_boundary)/np.maximum(distance,1e-9)
            # Simultaneous inward probes estimate only currently limiting
            # coordinates. The accepted target is checked with coupled dynamics.
            correction=np.where(response>1e-4,1.2*error/np.maximum(response,1e-4),.08)
            proposal=candidate.copy()
            proposal[active]-=direction[active]*np.clip(correction[active],.002,.4)
            proposal=np.clip(proposal,self.limits[:,0],self.limits[:,1])
            proposed_low,proposed_high=self._preview(proposal);calls+=1
            proposed_error=max(float(proposed_low.max()),float(proposed_high.max()))
            probe_error=max(float(probe_low.max()),float(probe_high.max()))
            if probe_error<proposed_error:
                proposal,proposed_low,proposed_high,proposed_error=probe,probe_low,probe_high,probe_error
            if proposed_error>=best_error:break
            candidate,low,high=proposal,proposed_low,proposed_high
            best,best_error=candidate.copy(),proposed_error
        change=np.abs(best-requested)
        return best,dict(enabled=True,preview_calls=calls,maximum_preview_calls=1+2*self.iterations,
            horizon_seconds=(self.steps+self.delay)*.002,measured_state_only=True,reference_preview_frames=0,
            maximum_application_delay_substeps=self.delay,application_schedules='immediate and maximum delay',
            native_preview_backend=self.native is not None,
            native_preview_threads=self.native.parallelism if self.native is not None else 1,
            predicted_limit_reserve_rad=self.reserve,predicted_limits_satisfied=bool(best_error<=0),
            original_predicted_excess_rad=max(0.,original_error),predicted_excess_rad=max(0.,best_error),
            changed_joints=int(np.count_nonzero(change>1e-9)),maximum_target_change_rad=float(change.max()),
            elapsed_ms=(time.perf_counter()-started)*1000,recursive_safety_guarantee=False,qualified=False)
