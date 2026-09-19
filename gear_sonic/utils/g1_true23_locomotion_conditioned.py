"""Full native leg pose corrections around the recovered locomotion network."""
from pathlib import Path
import numpy as np
import torch
from torch import nn
import yaml
from gear_sonic.utils.g1_true23_factory_native import NativeLegFactoryNetwork
from gear_sonic.utils.g1_true23_factory_locomotion import FactoryLocomotionTeleop,IDS

LOCOMOTION_CONDITIONED_WIDTH=1542  # native210 + commands8 + received1323 + closure1


class LocomotionConditionedActor(nn.Module):
    def __init__(self,weights,config,limits,neutral_recovery_gate=False):
        super().__init__()
        cfg=yaml.safe_load(Path(config).read_text(encoding='utf-8'))
        self.backbone=NativeLegFactoryNetwork(weights)
        self.neutral_recovery_gate=bool(neutral_recovery_gate)
        for p in self.backbone.parameters():p.requires_grad_(False)
        self.goal_head=nn.Sequential(nn.Linear(1020,256),nn.ELU(),nn.Linear(256,128),nn.ELU(),nn.Linear(128,12))
        nn.init.zeros_(self.goal_head[-1].weight);nn.init.zeros_(self.goal_head[-1].bias)
        self.register_buffer('default',torch.tensor(np.asarray(cfg['default_joint_q'])[IDS],dtype=torch.float32))
        self.register_buffer('factory_scale',torch.tensor(cfg['action_scale'][:12],dtype=torch.float32))
        self.register_buffer('upper_velocity_scale',torch.tensor((np.asarray(cfg['joint_kd'])/np.asarray(cfg['joint_kp']))[IDS][12:],dtype=torch.float32))
        self.register_buffer('limits',torch.tensor(limits,dtype=torch.float32));self.register_buffer('span',torch.ones(23))
        self.register_buffer('goal_mean',torch.zeros(1020));self.register_buffer('goal_scale',torch.ones(1020))

    def head_input(self,x):
        raw=self.backbone(x[:,:210].reshape(-1,5,42),x[:,210:218])
        # The balance core already observes applied-action history. Excluding
        # a second autoregressive action-history path from this correction head
        # removes the failure mode seen in the previous wrist-command learner.
        return torch.cat((x[:,218:1218],x[:,210:218],raw),1),raw

    def forward(self,x):
        features=x[:,218:1541];head,raw=self.head_input(x)
        delta=self.goal_head(((head-self.goal_mean)/self.goal_scale).clamp(-20,20))
        if self.neutral_recovery_gate:
            # Quiet, aligned neutral input retains the factory's standing
            # feedback. Any changed body intent or measured drift smoothly
            # restores the learned correction, including during input loss.
            yaw=torch.atan2(features[:,107],features[:,105]).abs()
            recovery=torch.stack((features[:,102:105].norm(dim=-1)/.08,yaw/.2,
                features[:,52:55].norm(dim=-1)/.2,features[:,46:49].norm(dim=-1)/.4),1).amax(1)
            gate=torch.maximum(x[:,1541],recovery).clamp(0,1)
            delta=delta*gate[:,None]
        legs=self.default[:12]+self.factory_scale*raw
        # Braking and held-pose recovery require corrections even after received
        # geometric motion closes. Zero initialization preserves the factory
        # function; a neutral packet must not disable subsequent feedback.
        legs=legs+delta*(self.limits[:12,1]-self.limits[:12,0])*.5
        upper=features[:,68:79]+self.upper_velocity_scale*features[:,91:102]
        requested=torch.cat((legs,upper),1)
        clipped=requested.clamp(self.limits[:,0]+.06,self.limits[:,1]-.06)
        target=requested+(clipped-requested).detach()
        return target-self.default

    def target(self,action):return (self.default+action).clamp(self.limits[:,0]+.06,self.limits[:,1]-.06)


class LocomotionConditionedController(FactoryLocomotionTeleop):
    def __init__(self,actor,firmware,contract,**options):
        import onnxruntime as ort
        from gear_sonic.utils.g1_true23_direct_body_goal import ReceivedBodyGoal
        lookahead_library=options.pop('lookahead_library',None);lookahead_steps=options.pop('lookahead_steps',30)
        response_library=options.pop('response_library',None)
        response_preserve_root=options.pop('response_preserve_root',False)
        planner_library=options.pop('planner_library',None)
        planner_checkpoint=options.pop('planner_checkpoint',None)
        if sum(bool(p) for p in (response_library,lookahead_library,planner_library))>1:raise ValueError('Select one native prediction controller')
        feedback=options.pop('lookahead_feedback',False)
        self.target_filter_alpha=float(options.pop('target_filter_alpha',1.))
        if not 0 < self.target_filter_alpha <= 1:raise ValueError('target_filter_alpha must be in(0,1]')
        self.has_applied_target=False
        super().__init__(firmware,contract,onnx_path=Path(firmware)/'human_loco_trainable_v1/factory_loco12.onnx',**options)
        self.lookahead=None
        self.response=None
        self.planner=None
        if planner_library:
            from gear_sonic.utils.g1_true23_feedback_planner import NativeFeedbackPlanner
            self.planner=NativeFeedbackPlanner(planner_library,options['model'],contract,self.kp,self.kd,options['tasks'],
                lookahead_steps,firmware=firmware,policy=self.policy,learned_checkpoint=planner_checkpoint,actor=actor)
        if response_library:
            from gear_sonic.utils.g1_true23_response_servo import NativeResponseServo
            self.response=NativeResponseServo(response_library,options['model'],contract,self.kp,self.kd,options['tasks'],lookahead_steps,response_preserve_root)
        if lookahead_library:
            if feedback:
                from gear_sonic.utils.g1_true23_feedback_shoot import NativeFeedbackLookahead
                self.lookahead=NativeFeedbackLookahead(lookahead_library,options['model'],contract,self.kp,self.kd,options['tasks'],lookahead_steps,
                    firmware=firmware,policy=self.policy)
            else:
                from gear_sonic.utils.g1_true23_native_shoot import NativePoseLookahead
                self.lookahead=NativePoseLookahead(lookahead_library,options['model'],contract,self.kp,self.kd,options['tasks'],lookahead_steps)
        self.body_goal=ReceivedBodyGoal(self.receiver.contract,task_closure_stand=True)
        opts=ort.SessionOptions();opts.intra_op_num_threads=1;opts.inter_op_num_threads=1
        self.session=ort.InferenceSession(str(actor),sess_options=opts,providers=['CPUExecutionProvider'])
        if self.session.get_inputs()[0].shape[-1]!=LOCOMOTION_CONDITIONED_WIDTH:raise ValueError('locomotion conditioned width mismatch')

    def command(self,qpos,qvel,now):
        from gear_sonic.utils.g1_true23_received_features import features_numpy
        from gear_sonic.utils.g1_true23_causal_controller import ControllerCommand
        qpos,qvel,rotation,_=self.prepare(qpos,qvel,now)
        self.policy.observation_command(qpos,qvel,-rotation[2],self.velocity)
        memory=self.receiver.history
        features=features_numpy(qpos,qvel,self.last_reference,len(self.last_reference['joint'])-1,
            self.receiver.contract['default_q'],memory.prior,memory.vector())
        alpha=self.body_goal.advance_blend(features,self.receiver)
        x=np.r_[self.policy.history.ravel(),self.policy.last_command,features,alpha].astype(np.float32)
        raw=self.session.run(['normalized_target'],{'features':x[None]})[0][0]
        if raw.shape!=(23,) or not np.isfinite(raw).all():raise ValueError('invalid conditioned locomotion target')
        target=np.clip(self.policy.default+raw,self.policy.limits[:,0]+.06,self.policy.limits[:,1]-.06)
        if self.lookahead:target=self.lookahead.refine(qpos,qvel,target,self.last_reference)
        if self.has_applied_target and self.target_filter_alpha<1:
            # The previous target comes from the actual plant history, including
            # when asynchronous computation misses an application substep.
            previous=self.policy.default[:12]+self.policy.previous
            target[:12]=previous+self.target_filter_alpha*(target[:12]-previous)
        if self.response:target=self.response.refine(qpos,qvel,target,self.last_reference)
        if self.planner:
            if self.receiver.gate.fault is not None:self.planner.clear_correction()
            else:target=self.planner.refine(qpos,qvel,target,self.last_reference,now)
        status=dict(self.status(),controller='native_locomotion_full_body_conditioned',motion_fraction=float(alpha))
        status['leg_target_filter_alpha']=self.target_filter_alpha
        if self.lookahead:status['native_lookahead']=dict(self.lookahead.status)
        if self.response:status['native_response']=dict(self.response.status)
        if self.planner:status['native_planner']=dict(self.planner.status)
        return ControllerCommand(target,status)

    def commit_applied(self,qpos,qvel,target):
        self.has_applied_target=True
        self.policy.previous[:]=np.asarray(target)[:12]-self.policy.default[:12]
        super().commit_applied(qpos,qvel,target)

    def import_native_history(self,control):
        super().import_native_history(control)
        self.has_applied_target=control>0
