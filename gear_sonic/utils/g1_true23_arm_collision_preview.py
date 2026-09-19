"""Two-frame reference-only collision projection with explicit 20ms preview."""
import numpy as np
import mujoco
from scipy.optimize import minimize

from gear_sonic.utils.g1_true23_arm_collision_projection import Native23CollisionArmProjector


class TwoFrameArmProjection:
    def __init__(self,model,step,*,hand_limit=.149,clearance=.002,max_iterations=60):
        self.parts=[Native23CollisionArmProjector(model,hand_limit=hand_limit,clearance=clearance) for _ in range(2)]
        self.step=np.asarray(step);self.max_iterations=max_iterations;self.hand_limit=hand_limit;self.clearance=clearance
        if not 0<hand_limit<=.15:raise ValueError("optimization margin must stay within original15cm task gate")
        self.qidx=self.parts[0].qidx;self.model=model
        roots={model.body(f"{side}_shoulder_pitch_link").id for side in ("left","right")}
        self.arm_bodies=set()
        for body in range(1,model.nbody):
            ancestor=body
            while ancestor:
                if ancestor in roots:self.arm_bodies.add(body);break
                ancestor=int(model.body_parentid[ancestor])

    def controllable(self,contact):
        return any(int(self.model.geom_bodyid[contact[key]]) in self.arm_bodies for key in ("geom1","geom2"))

    def block(self,part,pose,joints,pairs,goals):
        part.forward(pose,joints);hands,jac=part.hands()
        distances=np.array([mujoco.mj_geomDistance(self.model,part.data,a,b,.5,None) for a,b in pairs])
        derivative=np.empty((len(pairs),10));eps=1e-5
        for axis in range(10):
            x=joints.copy();x[axis]+=eps;part.forward(pose,x)
            plus=np.array([mujoco.mj_geomDistance(self.model,part.data,a,b,.5,None) for a,b in pairs])
            x[axis]-=2*eps;part.forward(pose,x)
            minus=np.array([mujoco.mj_geomDistance(self.model,part.data,a,b,.5,None) for a,b in pairs])
            derivative[:,axis]=(plus-minus)/(2*eps)
        error=hands-goals
        return np.r_[distances-self.clearance,(self.hand_limit-.0001)**2-np.sum(error**2,axis=1)],np.vstack((derivative,-2*np.einsum("si,sij->sj",error,jac)))

    def solve(self,poses,goals,previous,seed=None):
        references=np.asarray([p[self.qidx] for p in poses]);previous=np.asarray(previous)[self.qidx]
        low=np.tile(self.parts[0].ik.lower.ravel(),(2,1));high=np.tile(self.parts[0].ik.upper.ravel(),(2,1))
        low[0]=np.maximum(low[0],previous-self.step);high[0]=np.minimum(high[0],previous+self.step)
        if np.any(low>=high):return None,dict(passed=False,reason="empty_incoming_speed_box")
        x=np.clip(references if seed is None else seed,low,high)
        x[1]=np.clip(x[1],np.maximum(low[1],x[0]-self.step),np.minimum(high[1],x[0]+self.step))
        pairsets=[set(),set()]
        for i,part in enumerate(self.parts):
            for joints in (references[i],x[i]):
                part.forward(poses[i],joints)
                pairsets[i]|={tuple(sorted((c["geom1"],c["geom2"]))) for c in part.self_contacts() if self.controllable(c)}
        passes=[]
        for outer in range(3):
            pairs=[sorted(p) for p in pairsets];cache={}
            def evaluate(flat):
                if "x" not in cache or not np.array_equal(flat,cache["x"]):
                    q=flat.reshape(2,10);values=[];jacobians=[]
                    for i,part in enumerate(self.parts):
                        value,derivative=self.block(part,poses[i],q[i],pairs[i],goals[i])
                        jac=np.zeros((len(value),20));jac[:,i*10:(i+1)*10]=derivative
                        values.extend(value);jacobians.extend(jac)
                    delta=q[1]-q[0]
                    values.extend(self.step-delta);values.extend(self.step+delta)
                    jacobians.extend(np.c_[np.eye(10),-np.eye(10)]);jacobians.extend(np.c_[-np.eye(10),np.eye(10)])
                    cache.update(x=flat.copy(),values=np.asarray(values),jac=np.asarray(jacobians))
                return cache["values"],cache["jac"]
            def objective(flat):
                q=flat.reshape(2,10);error=q-references
                velocity_error=(q[1]-q[0])-(references[1]-references[0])
                cost=.5*np.sum(error**2)+.1*np.sum(velocity_error**2)
                gradient=error.copy();gradient[0]-=.2*velocity_error;gradient[1]+=.2*velocity_error
                return cost,gradient.ravel()
            result=minimize(lambda v:objective(v)[0],x.ravel(),jac=lambda v:objective(v)[1],
                            method="SLSQP",bounds=list(zip(low.ravel(),high.ravel())),
                            constraints=dict(type="ineq",fun=lambda v:evaluate(v)[0],jac=lambda v:evaluate(v)[1]),
                            options=dict(maxiter=self.max_iterations,ftol=1e-9,disp=False))
            x=result.x.reshape(2,10).copy()
            passes.append(dict(success=bool(result.success),message=str(result.message),iterations=int(result.nit),
                               constraint_min=float(np.min(evaluate(result.x)[0]))))
            added=False
            for i,part in enumerate(self.parts):
                part.forward(poses[i],x[i]);new={tuple(sorted((c["geom1"],c["geom2"]))) for c in part.self_contacts() if self.controllable(c)}
                added|=bool(new-pairsets[i]);pairsets[i]|=new
            if not added:break
        output=np.asarray(poses).copy();metrics=[]
        for i,part in enumerate(self.parts):
            output[i,self.qidx]=x[i];part.forward(poses[i],x[i]);hands,_=part.hands();contacts=part.self_contacts()
            errors=np.linalg.norm(hands-goals[i],axis=1)
            arm_contacts=[c for c in contacts if self.controllable(c)]
            fixed_contacts=[c for c in contacts if not self.controllable(c)]
            metrics.append(dict(hand_errors_m=errors.tolist(),hand_gate_pass=bool(np.all(errors<=.15)),
                                physical_hand_gate_m=.15,optimization_margin_m=self.hand_limit,
                                optimization_margin_pass=bool(np.all(errors<=self.hand_limit)),
                                self_contacts=contacts,collision_free=not contacts,
                                arm_contacts=arm_contacts,arm_collision_free=not arm_contacts,
                                unresolved_fixed_contacts=fixed_contacts))
        incoming=float(np.max(np.abs(x[0]-previous)/self.step));outgoing=float(np.max(np.abs(x[1]-x[0])/self.step))
        passed=all(m["hand_gate_pass"] and m["arm_collision_free"] for m in metrics) and max(incoming,outgoing)<=1+1e-10
        return output,dict(passed=passed,frames=metrics,incoming_80pct_speed_ratio=incoming,
                           pass_scope="arm_contacts_and_hand_targets_and_adjacent_speed_only",
                           all_self_collision_free=all(m["collision_free"] for m in metrics),
                           expected_next_80pct_speed_ratio=outgoing,optimizer_passes=passes,
                           joint_delta_max_rad=float(np.max(np.abs(x-references))))
