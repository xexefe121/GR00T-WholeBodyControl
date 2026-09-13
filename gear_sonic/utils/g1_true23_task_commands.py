"""Received task errors and learned factory command conditioning; sim only."""
from pathlib import Path
import numpy as np
import torch
from torch import nn
from gear_sonic.utils.g1_true23_locomotion_conditioned import LocomotionConditionedActor
from gear_sonic.utils.g1_true23_factory_locomotion import FactoryLocomotionTeleop

TASK_COMMAND_WIDTH=1582  # existing1542, task errors39, actual-history-valid1


def task_errors_numpy(model,data,qpos,qvel,reference,tasks):
    import mujoco
    data.qpos[:]=qpos;data.qvel[:]=qvel;mujoco.mj_kinematics(model,data)
    root=np.asarray(qpos[:3]);r={k:v[-1] for k,v in reference.items()}
    rotation=np.empty(9);mujoco.mju_quat2Mat(rotation,qpos[3:7]);rotation=rotation.reshape(3,3)
    yaw=np.arctan2(rotation[1,0],rotation[0,0]);c,s=np.cos(yaw),np.sin(yaw)
    heading=np.array([[c,-s,0],[s,c,0],[0,0,1.]])
    feet=data.xpos[[model.body(side+'_ankle_roll_link').id for side in ('left','right')]]
    actual=np.array([data.xpos[model.body(t['target_body']).id]
        +data.xmat[model.body(t['target_body']).id].reshape(3,3)@np.asarray(t['target_point']) for t in tasks])
    foot_error=(r['feet']-r['root'])-(feet-root)
    task_error=(r['tasks']-r['root'])-(actual-root)
    return np.r_[r['joint'][:12]-qpos[7:19],r['joint_velocity'][:12]-qvel[6:18],
        (foot_error@heading).ravel(),(task_error@heading).ravel()].astype(np.float32)


class TaskCommandActor(nn.Module):
    def __init__(self,weights,config,limits,base_checkpoint,full_body_corrections=False):
        super().__init__()
        self.full_body_corrections=bool(full_body_corrections)
        self.joint_command_count=23 if self.full_body_corrections else 12
        self.command_count=self.joint_command_count+5
        self.base=LocomotionConditionedActor(weights,config,limits)
        saved=torch.load(base_checkpoint,map_location='cpu',weights_only=False)
        if saved.get('request',{}).get('neutral_recovery_gate',False):raise ValueError('Requires the retained ungated native locomotion actor')
        self.base.load_state_dict(saved['actor'],strict=True)
        for p in self.base.parameters():p.requires_grad_(False)
        self.goal_head=nn.Sequential(nn.Linear(1059,256),nn.ELU(),nn.Linear(256,128),nn.ELU(),nn.Linear(128,self.command_count))
        nn.init.zeros_(self.goal_head[-1].weight);nn.init.zeros_(self.goal_head[-1].bias)
        self.register_buffer('default',self.base.default.clone())
        self.register_buffer('limits',self.base.limits.clone())
        self.register_buffer('span',torch.ones(23))
        self.register_buffer('goal_mean',torch.zeros(1059));self.register_buffer('goal_scale',torch.ones(1059))
        self.register_buffer('error_scale',torch.tensor([.15]*12+[.5]*12+[.12]*6+[.15]*6+[.1]*3))

    def head_input(self,x):
        h,raw=self.base.head_input(x[:,:1542])
        return torch.cat(((h-self.base.goal_mean)/self.base.goal_scale,x[:,1542:1581]/self.error_scale),1),raw

    def forward(self,x):
        h,raw=self.head_input(x)
        correction=self.goal_head(((h-self.goal_mean)/self.goal_scale).clamp(-20,20))
        return self.from_commands(x,correction,raw)

    def command_mean(self,x):
        """Trainable latent commands, before the frozen balance network.

        Sampling here can perturb foot clocks coherently. Adding independent
        noise to the final joint targets does not explore these commands.
        """
        h,_=self.head_input(x)
        return self.goal_head(((h-self.goal_mean)/self.goal_scale).clamp(-20,20))

    def from_commands(self,x,correction,raw=None):
        """Map joint corrections, two phases and three velocities to23 targets."""
        if raw is None:raw=self.base.backbone(x[:,:210].reshape(-1,5,42),x[:,210:218])
        command=x[:,210:218]
        n=self.joint_command_count
        phase=torch.pi*torch.tanh(correction[:,n:n+2])
        sine,cosine=command[:,:2],command[:,2:4]
        velocity=(command[:,5:]/.2+torch.tanh(correction[:,n+2:])*x.new_tensor([.6,.5,3.])).clamp(
            -x.new_tensor([1.2,1.,6.]),x.new_tensor([1.2,1.,6.]))
        changed=torch.cat((sine*torch.cos(phase)+cosine*torch.sin(phase),
            cosine*torch.cos(phase)-sine*torch.sin(phase),command[:,4:5],velocity*.2),1)
        changed_raw=self.base.backbone(x[:,:210].reshape(-1,5,42),changed)
        baseline=self.base.default+self.base(x[:,:1542])
        legs=baseline[:,:12]+self.base.factory_scale*(changed_raw-raw)
        legs=legs+correction[:,:12]*(self.limits[:12,1]-self.limits[:12,0])*.5
        upper=baseline[:,12:]
        if self.full_body_corrections:
            upper=upper+correction[:,12:23]*(self.limits[12:,1]-self.limits[12:,0])*.5
        requested=torch.cat((legs,upper),1)
        clipped=requested.clamp(self.limits[:,0]+.06,self.limits[:,1]-.06)
        target=requested+(clipped-requested).detach()
        previous=self.default[:12]+x[:,:210].reshape(-1,5,42)[:,-1,30:42]
        # Same alpha0.9 mapping as the preserved Pico demo. First warmed command
        # has no prior applied target; all other targets use actual plant history.
        alpha=1-.1*x[:,1581:1582]
        return torch.cat((previous+alpha*(target[:,:12]-previous),target[:,12:]),1)-self.default

    def target(self,action):return (self.default+action).clamp(self.limits[:,0]+.06,self.limits[:,1]-.06)


class TaskCommandController(FactoryLocomotionTeleop):
    def __init__(self,actor,firmware,contract,**options):
        import mujoco
        import onnxruntime as ort
        from gear_sonic.utils.g1_true23_direct_body_goal import ReceivedBodyGoal
        self.noise_seed=options.pop('noise_seed',None)
        self.noise_rng=np.random.default_rng(self.noise_seed) if self.noise_seed is not None else None
        self.feature_model=options['model'];self.feature_data=mujoco.MjData(self.feature_model)
        self.tasks=options['tasks'];self.has_applied_target=False
        self.last_observation_time=None
        super().__init__(firmware,contract,onnx_path=Path(firmware)/'human_loco_trainable_v1/factory_loco12.onnx',**options)
        self.body_goal=ReceivedBodyGoal(self.receiver.contract,task_closure_stand=True)
        opts=ort.SessionOptions();opts.intra_op_num_threads=1;opts.inter_op_num_threads=1
        self.session=ort.InferenceSession(str(actor),sess_options=opts,providers=['CPUExecutionProvider'])
        expected_width=TASK_COMMAND_WIDTH+(17 if self.noise_rng is not None else 0)
        if self.session.get_inputs()[0].shape[-1]!=expected_width:raise ValueError('Task-command input width mismatch')

    def observation(self,qpos,qvel,now):
        """Advance causal controller memory once and return the actor input."""
        from gear_sonic.utils.g1_true23_received_features import features_numpy
        if not np.isfinite(now) or (self.last_observation_time is not None and now<=self.last_observation_time):
            raise ValueError('Controller observation already advanced at this timestamp or clock regressed')
        self.last_observation_time=float(now)
        qpos,qvel,rotation,_=self.prepare(qpos,qvel,now)
        self.policy.observation_command(qpos,qvel,-rotation[2],self.velocity)
        memory=self.receiver.history
        f=features_numpy(qpos,qvel,self.last_reference,len(self.last_reference['joint'])-1,
            self.receiver.contract['default_q'],memory.prior,memory.vector())
        alpha=self.body_goal.advance_blend(f,self.receiver)
        errors=task_errors_numpy(self.feature_model,self.feature_data,qpos,qvel,self.last_reference,self.tasks)
        x=np.r_[self.policy.history.ravel(),self.policy.last_command,f,alpha,errors,float(self.has_applied_target)].astype(np.float32)
        if self.noise_rng is not None:x=np.r_[x,self.noise_rng.standard_normal(17)].astype(np.float32)
        return x

    def command(self,qpos,qvel,now):
        from gear_sonic.utils.g1_true23_causal_controller import ControllerCommand
        x=self.observation(qpos,qvel,now)
        raw=self.session.run(['normalized_target'],{'features':x[None]})[0][0]
        if raw.shape!=(23,) or not np.isfinite(raw).all():raise ValueError('Invalid task-command targets')
        margin=getattr(self,'target_margin',.06)
        target=np.clip(self.policy.default+raw,self.policy.limits[:,0]+margin,self.policy.limits[:,1]-margin)
        return ControllerCommand(target,dict(self.status(),controller='native23_received_task_commands',
            motion_fraction=float(x[1541]),leg_target_filter_alpha=.9,
            sampled_commands=self.noise_rng is not None,noise_seed=self.noise_seed))

    def commit_applied(self,qpos,qvel,target):
        self.has_applied_target=True;self.policy.previous[:]=target[:12]-self.policy.default[:12]
        super().commit_applied(qpos,qvel,target)

    def import_native_history(self,control):
        super().import_native_history(control);self.has_applied_target=control>0
