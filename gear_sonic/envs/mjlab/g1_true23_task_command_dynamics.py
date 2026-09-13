"""Full-body task-error observations for native23 command-conditioned dynamics."""
import torch
from gear_sonic.envs.mjlab.g1_true23_locomotion_conditioned_dynamics import LocomotionConditionedEnv
from gear_sonic.envs.mjlab.g1_true23_causal_dynamics import quaternion_matrix


class TaskCommandEnv(LocomotionConditionedEnv):
    def __init__(self,*args,**kwargs):
        self.task_commands_ready=False
        super().__init__(*args,**kwargs)
        self.task_commands_ready=True

    def observe(self):
        x=super().observe()
        if not self.task_commands_ready:return x
        index=torch.minimum(self.frames,self.lengths[self.clips]-1)
        r={key:value[self.clips,index] for key,value in self.references.items()}
        rotation=quaternion_matrix(self.q[:,3:7]);yaw=torch.atan2(rotation[:,1,0],rotation[:,0,0])
        c,s=torch.cos(yaw),torch.sin(yaw);z=torch.zeros_like(c);one=torch.ones_like(c)
        h=torch.stack((c,-s,z,s,c,z,z,z,one),1).reshape(-1,3,3)
        root=self.q[:,:3]
        feet=self.sim.data.xpos[:,self.feet]
        mats=self.sim.data.xmat[:,self.task_ids].reshape(self.count,3,3,3)
        tasks=self.sim.data.xpos[:,self.task_ids]+torch.einsum('ntij,tj->nti',mats,self.task_offsets)
        foot=(r['feet']-r['root'][:,None])-(feet-root[:,None])
        task=(r['tasks']-r['root'][:,None])-(tasks-root[:,None])
        errors=torch.cat((r['joint'][:,:12]-self.q[:,7:19],r['joint_velocity'][:,:12]-self.v[:,6:18],
            torch.einsum('nti,nij->ntj',foot,h).flatten(1),torch.einsum('nti,nij->ntj',task,h).flatten(1)),1)
        valid=(~self.canonical_world)|(self.age>0)
        return torch.cat((x,errors,valid[:,None].float()),1)
