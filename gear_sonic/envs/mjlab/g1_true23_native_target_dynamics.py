"""Original native benchmark actuation with factory-policy history conversion."""
import torch
from gear_sonic.envs.mjlab.g1_true23_task_command_dynamics import TaskCommandEnv
from gear_sonic.utils.g1_true23_factory_policy import IDS


class NativeTargetEnv(TaskCommandEnv):
    def __init__(self,*args,**kwargs):
        self.native_target_ready=False
        super().__init__(*args,**kwargs)
        self.factory_kp=self.kp.clone();self.factory_kd=self.kd.clone()
        self.kp=self.old_kp.clone();self.kd=self.old_kd.clone()
        self.native_target_ready=True
        self.reset(torch.arange(self.count,device=self.device))

    def native_prop(self):
        prop=super().native_prop()
        if not self.native_target_ready:return prop
        q,v=self.q[:,7:19],self.v[:,6:18]
        previous=self.factory_default[:12]+self.loco_previous
        equivalent=q+(self.kp[:12]*(previous-q)+(self.factory_kd[:12]-self.kd[:12])*v)/self.factory_kp[:12]
        prior=equivalent-self.factory_default[:12]
        # A fresh actual-start episode has no applied command yet.
        prior=torch.where((self.canonical_world&(self.age==0))[:,None],0.,prior)
        return torch.cat((prop[:,:30],prior.clamp(-10,10)),1)

    def restore(self,ids,rows,clip_override=None):
        if not self.native_target_ready:return super().restore(ids,rows,clip_override)
        self.clips[ids]=rows[:,0].long() if clip_override is None else clip_override
        self.frames[ids]=rows[:,1].long()
        self.q[ids],self.v[ids]=rows[:,2:32],rows[:,32:61]
        previous=self.default+rows[:,61:84]*self.old_scale
        self.reset_teacher[ids]=self.teacher_table[rows[:,0].long(),rows[:,1].long()].clamp(self.limits[:,0],self.limits[:,1])
        self.factory_prior[ids]=0
        self.factory_prior[ids[:,None],self.q.new_tensor(IDS,dtype=torch.long)[None]]=(previous-self.factory_default)/.25

    def control_torque(self,targets):
        if not self.native_target_ready:return super().control_torque(targets)
        if self.command_delay_substeps:
            delayed=self.control_substep<self.delay_steps
            targets=torch.where(delayed[:,None],self.held_for_delay,targets)
            self.control_substep+=1
        return (self.kp*(targets-self.q[:,7:])-self.kd*self.v[:,6:]).clamp(-self.effort,self.effort)
