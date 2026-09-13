"""Bounded reference-only arm projection against active native self collisions."""
import mujoco
import numpy as np
from scipy.optimize import minimize

from gear_sonic.utils.g1_true23_intent_arm_ik import Native23ArmIK,HAND_POINTS_LOCAL


class Native23CollisionArmProjector:
    def __init__(self,model,*,clearance=.002,hand_limit=.15,max_iterations=80,collision_jacobian="finite_distance"):
        self.model=model;self.data=mujoco.MjData(model);self.ik=Native23ArmIK(model)
        self.qidx=self.ik.qpos_indices.ravel();self.vidx=self.ik.dof_indices.ravel()
        self.clearance=clearance;self.hand_limit=hand_limit;self.max_iterations=max_iterations
        self.collision_jacobian=collision_jacobian
        if collision_jacobian not in ("finite_distance","contact_point"):
            raise ValueError("unknown collision Jacobian")

    def forward(self,pose,joints):
        self.data.qpos[:]=pose;self.data.qpos[self.qidx]=joints;self.data.qvel[:]=0.
        mujoco.mj_fwdPosition(self.model,self.data)

    def self_contacts(self):
        result=[]
        for contact in self.data.contact:
            a,b=int(contact.geom1),int(contact.geom2)
            if self.model.geom_bodyid[a] and self.model.geom_bodyid[b] and contact.dist<0:
                result.append(dict(geom1=a,geom2=b,distance_m=float(contact.dist),
                    body1=self.model.body(int(self.model.geom_bodyid[a])).name,
                    body2=self.model.body(int(self.model.geom_bodyid[b])).name,
                    exclude=int(contact.exclude),constraint_address=int(contact.efc_address),
                    constraint_dimension=int(contact.dim)))
        return result

    def distance_gradient(self,a,b):
        segment=np.empty(6)
        distance=float(mujoco.mj_geomDistance(self.model,self.data,a,b,.5,segment))
        gradient=np.zeros(10)
        if abs(distance)<1e-10 or distance>=.5:
            return distance,gradient
        normal=(segment[3:]-segment[:3])/distance
        ja,jb=np.zeros((3,self.model.nv)),np.zeros((3,self.model.nv))
        mujoco.mj_jac(self.model,self.data,ja,None,segment[:3],int(self.model.geom_bodyid[a]))
        mujoco.mj_jac(self.model,self.data,jb,None,segment[3:],int(self.model.geom_bodyid[b]))
        gradient=normal@(jb-ja)[:,self.vidx]
        return distance,gradient

    def hands(self):
        values=[];jac=[]
        for side,body in enumerate(self.ik.body_ids):
            point=self.data.xpos[body]+self.data.xmat[body].reshape(3,3)@HAND_POINTS_LOCAL[side]
            jp=np.zeros((3,self.model.nv));mujoco.mj_jac(self.model,self.data,jp,None,point,int(body))
            values.append(point.copy());jac.append(jp[:,self.vidx])
        return np.array(values),np.array(jac)

    def solve(self,pose,hand_goals,*,previous_pose=None,maximum_step=None,seed_pose=None):
        pose=np.asarray(pose).copy();initial=pose[self.qidx].copy()
        lower,upper=self.ik.lower.ravel().copy(),self.ik.upper.ravel().copy()
        if previous_pose is not None:
            if maximum_step is None:raise ValueError("previous pose requires step limits")
            lower=np.maximum(lower,np.asarray(previous_pose)[self.qidx]-maximum_step)
            upper=np.minimum(upper,np.asarray(previous_pose)[self.qidx]+maximum_step)
        if np.any(lower>=upper):raise ValueError("empty native arm interval")
        self.forward(pose,np.clip(initial,lower,upper))
        pairs={tuple(sorted((c["geom1"],c["geom2"]))) for c in self.self_contacts()}
        seed=initial if seed_pose is None else np.asarray(seed_pose)[self.qidx]
        joints=np.clip(seed,lower,upper);self.forward(pose,joints)
        pairs|={tuple(sorted((c["geom1"],c["geom2"]))) for c in self.self_contacts()}
        iterations=[]
        for outer in range(3):
            pairs_sorted=sorted(pairs);cache={}
            def evaluate(x):
                if "x" not in cache or not np.array_equal(x,cache["x"]):
                    self.forward(pose,x);hand,jac=self.hands()
                    values=[];grads=[]
                    distances=np.array([mujoco.mj_geomDistance(self.model,self.data,a,b,.5,None) for a,b in pairs_sorted])
                    if self.collision_jacobian=="finite_distance":
                        collision_jac=np.empty((len(pairs_sorted),10));epsilon=1e-5
                        for axis in range(10):
                            perturbed=x.copy();perturbed[axis]+=epsilon;self.forward(pose,perturbed)
                            plus=np.array([mujoco.mj_geomDistance(self.model,self.data,a,b,.5,None) for a,b in pairs_sorted])
                            perturbed[axis]-=2*epsilon;self.forward(pose,perturbed)
                            minus=np.array([mujoco.mj_geomDistance(self.model,self.data,a,b,.5,None) for a,b in pairs_sorted])
                            collision_jac[:,axis]=(plus-minus)/(2*epsilon)
                        self.forward(pose,x)
                    else:
                        collision_jac=np.array([self.distance_gradient(a,b)[1] for a,b in pairs_sorted])
                    values.extend(distances-self.clearance);grads.extend(collision_jac)
                    error=hand-hand_goals
                    # Numerical optimization margin is inside the unchanged
                    # physical task gate; the reported pass uses <=.15 exactly.
                    values.extend((self.hand_limit-.0001)**2-np.sum(error**2,axis=1))
                    grads.extend(-2*np.einsum("si,sij->sj",error,jac))
                    cache.update(x=x.copy(),value=np.asarray(values),jac=np.asarray(grads))
                return cache["value"],cache["jac"]
            result=minimize(lambda x:.5*np.sum((x-initial)**2),joints,jac=lambda x:x-initial,
                            bounds=list(zip(lower,upper)),constraints=dict(type="ineq",fun=lambda x:evaluate(x)[0],jac=lambda x:evaluate(x)[1]),
                            method="SLSQP",options=dict(maxiter=self.max_iterations,ftol=1e-10,disp=False))
            joints=result.x.copy();self.forward(pose,joints);contacts=self.self_contacts()
            new_pairs={tuple(sorted((c["geom1"],c["geom2"]))) for c in contacts}
            iterations.append(dict(success=bool(result.success),status=int(result.status),message=str(result.message),
                                   iterations=int(result.nit),constraint_min=float(np.min(evaluate(joints)[0]))))
            if not new_pairs-pairs:break
            pairs|=new_pairs
        self.forward(pose,joints);hand,_=self.hands();contacts=self.self_contacts()
        output=pose.copy();output[self.qidx]=joints
        errors=np.linalg.norm(hand-hand_goals,axis=1)
        return output,dict(collision_free=not contacts,remaining_contacts=contacts,original_relative_hand_errors_m=errors.tolist(),
                           hand_gate_pass=bool(np.all(errors<=self.hand_limit)),joint_delta_max_rad=float(np.max(np.abs(joints-initial))),
                           collision_jacobian=self.collision_jacobian,hand_jacobian="analytic_native_point_jacobian",
                           native_range_pass=bool(np.all((joints>=lower-1e-8)&(joints<=upper+1e-8))),optimizer_passes=iterations,
                           previous_step_bound_applied=previous_pose is not None,
                           supplied_seed=seed_pose is not None,
                           step_ratio_max=None if previous_pose is None else float(np.max(np.abs(joints-np.asarray(previous_pose)[self.qidx])/maximum_step)))
