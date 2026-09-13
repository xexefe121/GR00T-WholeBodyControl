"""Continuous target correction using native contact-response predictions."""
import ctypes as ct
import time
import numpy as np
import mujoco
from scipy.optimize import lsq_linear
from scipy.spatial.transform import Rotation

D=ct.POINTER(ct.c_double)
I=ct.POINTER(ct.c_int)
def ptr(a):return a.ctypes.data_as(D)

class NativeResponseServo:
    def __init__(self,library,model,contract,kp,kd,tasks,steps=10,preserve_root=False):
        if mujoco.__version__!='3.2.3':raise ValueError('Response servo requires native MuJoCo3.2.3')
        self.model=model;self.steps=int(steps)
        if not 1<=self.steps<=100:raise ValueError('Prediction steps must be1..100')
        self.limits=np.asarray(contract['joint_limits']);self.lower=self.limits[:,0]+.06;self.upper=self.limits[:,1]-.06
        self.ids=np.array([model.body(s+'_ankle_roll_link').id for s in ('left','right')]+
            [model.body(t['target_body']).id for t in tasks],np.int32)
        offsets=np.array([[0,0,0],[0,0,0]]+[t['target_point'] for t in tasks],float)
        arrays=[np.ascontiguousarray(a,float) for a in (kp,kd,contract['native_effort'],contract['native_velocity'],self.limits,offsets)]
        self.lib=ct.CDLL(str(library))
        self.lib.response_create.argtypes=[ct.c_void_p,*[D]*5,I,D];self.lib.response_create.restype=ct.c_void_p
        self.lib.response_destroy.argtypes=[ct.c_void_p]
        self.lib.response_predict.argtypes=[ct.c_void_p,D,D,D,ct.c_int,ct.c_int,D,I];self.lib.response_predict.restype=ct.c_int
        self.context=self.lib.response_create(model._address,*[ptr(a) for a in arrays[:5]],self.ids.ctypes.data_as(I),ptr(arrays[5]))
        if not self.context:raise RuntimeError('Native response context failed')
        self.status={};self.regularization=.3;self.preserve_root=bool(preserve_root)

    def predict(self,q,v,targets):
        q,v,targets=[np.ascontiguousarray(a,float) for a in (q,v,targets)]
        out=np.empty((len(targets),89));failed=np.empty(len(targets),np.int32)
        if self.lib.response_predict(self.context,ptr(q),ptr(v),ptr(targets),len(targets),self.steps,ptr(out),failed.ctypes.data_as(I)):
            raise RuntimeError('Native response prediction failed')
        if not np.isfinite(out).all():raise RuntimeError('Nonfinite native prediction')
        return out,failed.astype(bool)

    def residual(self,out,r):
        h=self.steps*.002
        q,v=out[:,:30],out[:,30:59]
        point,point_v=out[:,59:74].reshape(-1,5,3),out[:,74:].reshape(-1,5,3)
        desired_root=r['root']+h*r['root_velocity']
        desired_point=np.concatenate((r['feet'],r['tasks']))+h*np.concatenate((r['feet_velocity'],r['task_velocity']))
        desired_point_v=np.concatenate((r['feet_velocity'],r['task_velocity']))
        root=(q[:,:3]-desired_root+.1*(v[:,:3]-r['root_velocity']))/.2
        actual=Rotation.from_quat(q[:,[4,5,6,3]])
        desired=Rotation.from_rotvec(h*r['root_omega'])*Rotation.from_matrix(r['root_rotation'])
        angular=((actual*desired.inv()).as_rotvec()+.1*(actual.apply(v[:,3:6])-r['root_omega']))/.26
        joints=((q[:,7:]-(r['joint']+h*r['joint_velocity'])+.1*(v[:,6:]-r['joint_velocity']))/
            np.r_[np.full(12,.15),np.full(11,.25)])
        relative=(point-q[:,None,:3])-(desired_point-desired_root)
        relative+=.1*((point_v-v[:,None,:3])-(desired_point_v-r['root_velocity']))
        relative/=np.array([.12,.12,.15,.15,.1])[None,:,None]
        return np.concatenate((root,angular,joints,relative.reshape(len(out),-1)),axis=1)

    def refine(self,q,v,baseline,reference):
        tick=time.perf_counter_ns();r={k:a[-1] for k,a in reference.items()}
        baseline=np.clip(baseline,self.lower,self.upper)
        candidates=np.repeat(baseline[None],47,axis=0)
        for j in range(23):
            candidates[1+j,j]=min(self.upper[j],baseline[j]+.01)
            candidates[24+j,j]=max(self.lower[j],baseline[j]-.01)
        out,failed=self.predict(q,v,candidates);errors=self.residual(out,r)
        jac=((errors[1:24]-errors[24:47])/(candidates[1:24]-candidates[24:47]).diagonal()[:,None]).T
        jac[:,failed[1:24]|failed[24:47]]=0.
        # Full native target interval; regularization asks the factory policy
        # to supply balance where received tracking cannot improve prediction.
        penalty=np.sqrt(self.regularization)/.25
        matrix=np.r_[jac,penalty*np.eye(23)];goal=np.r_[-errors[0],np.zeros(23)]
        if self.preserve_root:
            import clarabel
            from scipy import sparse
            root_jac=((out[1:24,30:36]-out[24:47,30:36])/
                (candidates[1:24]-candidates[24:47]).diagonal()[:,None]).T
            root_jac[:,failed[1:24]|failed[24:47]]=0.
            settings=clarabel.DefaultSettings();settings.verbose=False;settings.max_iter=40
            settings.tol_feas=settings.tol_gap_abs=settings.tol_gap_rel=1e-7
            bounds=np.r_[np.zeros(6),self.upper-baseline,baseline-self.lower]
            constraints=np.r_[root_jac,np.eye(23),-np.eye(23)]
            fit=clarabel.DefaultSolver(sparse.csc_matrix(np.triu(matrix.T@matrix)),-matrix.T@goal,
                sparse.csc_matrix(constraints),bounds,[clarabel.ZeroConeT(6),clarabel.NonnegativeConeT(46)],settings).solve()
            delta=np.asarray(fit.x);success=str(fit.status) in ('Solved','AlmostSolved') and np.isfinite(delta).all()
            success=bool(success and np.max(np.abs(root_jac@delta))<1e-4)
        else:
            fit=lsq_linear(matrix,goal,bounds=(self.lower-baseline,self.upper-baseline),method='bvls',tol=1e-6,max_iter=40)
            delta=fit.x;success=bool(fit.success)
        fractions=np.array([0.,.125,.25,.5,1.])
        proposals=np.clip(baseline+fractions[:,None]*delta,self.lower,self.upper)
        predicted,physical_bad=self.predict(q,v,proposals);e=self.residual(predicted,r)
        bad=physical_bad.copy()
        root_changes=predicted[:,30:36]-predicted[0,30:36]
        if self.preserve_root:
            # A nonlinear query must confirm the linear equality before a
            # correction can alter the real target. No base force is applied.
            bad|=(np.abs(root_changes)>np.array([.01]*3+[.03]*3)).any(axis=1)
        costs=np.sum(e*e,axis=1)+self.regularization*np.sum(((proposals-baseline)/.25)**2,axis=1)
        costs[bad]=np.inf
        selected=int(np.argmin(costs)) if np.isfinite(costs).any() and success else 0
        self.status=dict(prediction_steps=self.steps,prediction_seconds=self.steps*.002,
            differentiated_targets=23,queries=52,selected_fraction=float(fractions[selected]),
            predicted_cost=float(costs[selected]) if np.isfinite(costs[selected]) else None,
            baseline_predicted_cost=float(costs[0]) if np.isfinite(costs[0]) else None,
            solver_succeeded=success,all_proposals_predicted_failure=bool(physical_bad.all()),
            all_proposals_rejected=bool(bad.all()),preserve_factory_base_motion=self.preserve_root,
            selected_base_velocity_change=root_changes[selected].tolist(),
            response_ms=(time.perf_counter_ns()-tick)*1e-6,future_reference_frames=0)
        return proposals[selected].copy()

    def close(self):
        if self.context:self.lib.response_destroy(self.context);self.context=None
