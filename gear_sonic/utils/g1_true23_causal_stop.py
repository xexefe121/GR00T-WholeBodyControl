"""Causal, generated two-second return-to-standing reference after input loss."""
import numpy as np
import mujoco
from scipy.spatial.transform import Rotation, Slerp


class StandingReference:
    def __init__(self,model,first,standing_qpos,tasks):
        self.model=model;self.data=mujoco.MjData(model);self.tasks=tasks;self.step=0
        self.first=np.r_[first['body_pos_w'][0],first['body_quat_w'][0],first['joint_pos']]
        self.last=np.asarray(standing_qpos).copy()
        self.last[:2]=self.first[:2]
        q=self.first[3:7];w,x,y,z=q
        yaw=np.arctan2(2*(w*z+x*y),1-2*(y*y+z*z))
        self.last[3:7]=[np.cos(yaw/2),0,0,np.sin(yaw/2)]
        self.slerp=Slerp([0.,1.],Rotation.from_quat(np.stack((self.first[3:7],self.last[3:7]))[:,[1,2,3,0]]))
        self.previous=first

    def next(self):
        self.step+=1;u=min(self.step/100,1.);weight=u*u*u*(10-15*u+6*u*u)
        pose=self.first+(self.last-self.first)*weight
        pose[3:7]=self.slerp(weight).as_quat()[[3,0,1,2]]
        self.data.qpos[:]=pose;mujoco.mj_kinematics(self.model,self.data)
        fields=dict(joint_pos=pose[7:].copy(),joint_vel=(pose[7:]-self.previous['joint_pos'])/.02,
            body_pos_w=self.data.xpos[1:].copy(),body_quat_w=self.data.xquat[1:].copy(),
            body_lin_vel_w=(self.data.xpos[1:]-self.previous['body_pos_w'])/.02,body_ang_vel_w=np.zeros((24,3)))
        prior=Rotation.from_quat(self.previous['body_quat_w'][:,[1,2,3,0]])
        current=Rotation.from_quat(fields['body_quat_w'][:,[1,2,3,0]])
        fields['body_ang_vel_w']=(current*prior.inv()).as_rotvec()/.02
        positions=[];quats=[]
        for task in self.tasks:
            i=self.model.body(task['target_body']).id
            positions.append(self.data.xpos[i]+self.data.xmat[i].reshape(3,3)@task['target_point'])
            quats.append(self.data.xquat[i].copy())
        self.previous=fields
        return fields,np.asarray(positions),np.asarray(quats)
