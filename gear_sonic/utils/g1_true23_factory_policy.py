"""Trainable received-pose native23 policy initialized from factory balance.

All recovered network layers remain trainable. A zero-initialized body-error
input supplies position feedback absent from the original dance actor. This
module does not load motion archives or communicate with hardware.
"""
from pathlib import Path
import numpy as np
import torch
from torch import nn

IDS = list(range(13)) + list(range(15,20)) + list(range(22,27))
WIDTH = 655  # memory465, proprio93, current quaternion4, target71, errors22


def linear_chain(weights, prefix, indices, final_tanh=False):
    layers = []
    for i in indices:
        weight = weights[f'{prefix}.net.net.{i}.weight']
        bias = weights[f'{prefix}.net.net.{i}.bias']
        layer = nn.Linear(weight.shape[1], weight.shape[0])
        with torch.no_grad():
            layer.weight.copy_(torch.from_numpy(weight.copy()))
            layer.bias.copy_(torch.from_numpy(bias.copy()))
        layers.append(layer)
        if i != indices[-1]: layers.append(nn.ELU())
    if final_tanh: layers.append(nn.Tanh())
    return nn.Sequential(*layers)


class FactoryReceivedActor(nn.Module):
    def __init__(self, onnx_path: Path, default29, limits, margin=.06,heading_relative=False):
        super().__init__()
        import onnx
        graph = onnx.load(str(onnx_path))
        w = {v.name:onnx.numpy_helper.to_array(v) for v in graph.graph.initializer}
        self.memory = linear_chain(w,'MemoryEncoder',[0,2,4],True)
        self.estimator = linear_chain(w,'StateEstimator',[0,2,4])
        self.actor = linear_chain(w,'Actor',[0,2,4,6])
        self.error_input = nn.Linear(22,512,bias=False)
        nn.init.zeros_(self.error_input.weight)
        for key,name in [('memory_mean','MemoryEncoder.norm_layer._mean'),('memory_scale','onnx::Div_75'),
                         ('actor_mean','Actor.norm_layer._mean'),('actor_scale','onnx::Div_76')]:
            self.register_buffer(key,torch.from_numpy(w[name].copy()))
        self.register_buffer('ids',torch.tensor(IDS,dtype=torch.long))
        self.register_buffer('default',torch.tensor(np.asarray(default29)[IDS],dtype=torch.float32))
        self.register_buffer('limits',torch.tensor(limits,dtype=torch.float32))
        self.register_buffer('span',torch.full((23,),.25))
        self.margin = margin
        self.heading_relative = heading_relative

    @staticmethod
    def product(a,b):
        aw,ax,ay,az=a.unbind(-1);bw,bx,by,bz=b.unbind(-1)
        return torch.stack((aw*bw-ax*bx-ay*by-az*bz,aw*bx+ax*bw+ay*bz-az*by,
            aw*by-ax*bz+ay*bw+az*bx,aw*bz+ax*by-ay*bx+az*bw),-1)

    def relative_frame(self,quat_xyzw,target,errors):
        q=quat_xyzw[:,[3,0,1,2]]
        w,x,y,z=q.unbind(-1)
        yaw=torch.atan2(2*(w*z+x*y),1-2*(y*y+z*z))
        zero=torch.zeros_like(yaw)
        inverse=torch.stack((torch.cos(yaw*.5),zero,zero,-torch.sin(yaw*.5)),-1)
        tilt=self.product(inverse,q)
        desired=self.product(inverse,target[:,3:7][:,[3,0,1,2]])
        desired=desired*torch.where(desired[:,:1]<0,-1.,1.)
        # Rotate current-body displacement into current-heading coordinates.
        w,x,y,z=tilt.unbind(-1)
        rotation=torch.stack((1-2*(y*y+z*z),2*(x*y-w*z),2*(x*z+w*y),
            2*(x*y+w*z),1-2*(x*x+z*z),2*(y*z-w*x),
            2*(x*z-w*y),2*(y*z+w*x),1-2*(x*x+y*y)),-1).reshape(-1,3,3)
        displacement=torch.bmm(rotation,(errors[:,:3]*.2).unsqueeze(-1)).squeeze(-1)
        target=torch.cat((displacement[:,:2],target[:,2:3],desired[:,[1,2,3,0]],target[:,7:]),-1)
        return tilt[:,[1,2,3,0]],target

    def raw29(self, x):
        memory, proprio, quat, target, errors = x[:,:465],x[:,465:558],x[:,558:562],x[:,562:633],x[:,633:]
        if self.heading_relative:quat,target=self.relative_frame(quat,target,errors)
        encoded = self.memory((memory-self.memory_mean)/self.memory_scale)
        velocity = self.estimator(encoded)
        joined = torch.cat((velocity,encoded,proprio,quat,target),-1)
        hidden = self.actor[0]((joined-self.actor_mean)/self.actor_scale) + self.error_input(errors)
        for layer in self.actor[1:]: hidden = layer(hidden)
        return hidden

    def forward(self, x):
        raw = self.raw29(x)[:,self.ids]
        lower = (self.limits[:,0]+self.margin-self.default)/self.span
        upper = (self.limits[:,1]-self.margin-self.default)/self.span
        projected = raw.clamp(lower,upper)
        # Legal distribution means with a declared straight-through gradient.
        # This changes commands only; the simulator never projects robot state.
        return raw + (projected-raw).detach()

    def target(self, actions):
        return (self.default+self.span*actions).clamp(self.limits[:,0]+self.margin,self.limits[:,1]-self.margin)


def body_errors_numpy(qpos,qvel,feet,tasks,goal_root,goal_feet,goal_tasks):
    from gear_sonic.utils.g1_true23_factory_controller import quaternion_matrix
    r = quaternion_matrix(qpos[3:7])
    root = qpos[:3]
    hand_scales = np.array([.15,.15,.10])[:,None]
    return np.concatenate(((goal_root-root)@r/.2,qvel[:3]@r/.3,
        (((goal_feet-goal_root)-(feet-root))@r/.12).ravel(),
        ((((goal_tasks-goal_root)-(tasks-root))@r)/hand_scales).ravel(),[qpos[2]/.8])).astype(np.float32)


class FactoryReceivedController:
    """Export runtime: measured state + current/past received reference only."""
    def __init__(self, actor_path, config_path, joint_limits):
        # Reuse the verified raw observation and applied-command convention.
        import yaml
        import onnxruntime as ort
        cfg = yaml.safe_load(Path(config_path).read_text())
        self.default = np.asarray(cfg['default_dof_pos'],np.float32)[IDS]
        self.kp = np.asarray(cfg['kp'])[IDS]
        self.kd = np.asarray(cfg['kd'])[IDS]
        self.limits = np.asarray(joint_limits)
        self.previous = np.zeros(29,np.float32)
        self.history = np.zeros((5,93),np.float32)
        self.initialized = False
        opts = ort.SessionOptions(); opts.intra_op_num_threads=1; opts.inter_op_num_threads=1
        self.session = ort.InferenceSession(str(actor_path),sess_options=opts,providers=['CPUExecutionProvider'])
        if self.session.get_inputs()[0].shape[-1] != WIDTH: raise ValueError('factory received input width mismatch')

    def step(self,qpos,qvel,target_state,feet,tasks,goal_root,goal_feet,goal_tasks):
        from gear_sonic.utils.g1_true23_factory_controller import quaternion_matrix,normalized_quaternion
        q29,v29 = np.zeros(29,np.float32),np.zeros(29,np.float32)
        q29[IDS],v29[IDS] = qpos[7:],qvel[6:]
        r = quaternion_matrix(qpos[3:7])
        proprio = np.r_[qvel[3:6],-r[2],q29,v29,self.previous].astype(np.float32)
        if not self.initialized: self.history[:] = proprio; self.initialized = True
        else: self.history[-1]=proprio
        errors = body_errors_numpy(qpos,qvel,feet,tasks,goal_root,goal_feet,goal_tasks)
        x = np.r_[self.history.ravel(),proprio,normalized_quaternion(qpos[3:7])[[1,2,3,0]],target_state,errors].astype(np.float32)
        if x.shape != (WIDTH,) or not np.isfinite(x).all(): raise ValueError('invalid received factory features')
        action = self.session.run(['normalized_target'],{'features':x[None]})[0][0]
        if action.shape != (23,) or not np.isfinite(action).all(): raise ValueError('invalid factory target')
        return np.clip(self.default+.25*action,self.limits[:,0]+.06,self.limits[:,1]-.06)

    def commit_applied(self,target):
        # Only a physical control boundary advances observed history. Repeated
        # or late proposals cannot create fictitious past observations.
        self.history[:-1]=self.history[1:]
        self.previous[:] = 0
        self.previous[IDS] = (np.asarray(target)-self.default)/.25
