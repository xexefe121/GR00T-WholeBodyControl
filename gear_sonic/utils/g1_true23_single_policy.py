"""Trainable native23 balance actor with causal full-body conditioning.

One actor handles all phases. Its initial function preserves the established
neutral BFM controller. All BFM actor weights train through dynamics; the
reference encoder and observation normalizers remain frozen. A zero-initialized
goal adapter receives every causal full-body feature, and a zero-initialized
output adapter makes the whole native target interval reachable. There is no
separate motion controller, phase selector, or bounded 0.15-rad residual.
"""
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from gear_sonic.utils.g1_true23_bfmzero_inference import BFMZeroInference,reference_features


def smooth_native_fraction(raw, tau):
    """Smooth [0, 1] saturation with polynomial, ordinary-derivative tails.

    This is s(raw)-s(raw-1), where s(x)=(x+sqrt(x*x+tau*tau))/2.
    The symmetric form avoids subtracting two large positive values. The
    algebraic tail retains useful gradients outside the native interval.
    """
    value=raw.double()
    def positive(x):
        radius=torch.sqrt(x.square()+tau*tau)
        return torch.where(x>=0,.5*(x+radius),.5*tau*tau/(radius-x).clamp_min(tau))
    lower=positive(value)-positive(value-1)
    upper=1-positive(1-value)+positive(-value)
    return torch.where(value<=.5,lower,upper).to(raw.dtype)


class SinglePolicy(nn.Module):
    def __init__(self,weights,contract,neutral_motion,feature_mean,feature_scale,*,smooth_action_tau=0.,normalize_range_adapter=False):
        super().__init__()
        if not np.isfinite(smooth_action_tau) or smooth_action_tau<0:
            raise ValueError('smooth action temperature must be finite and nonnegative')
        self.smooth_action_tau=float(smooth_action_tau)
        self.normalize_range_adapter=bool(normalize_range_adapter)
        loaded=BFMZeroInference(weights).weights
        self.actor_weights=nn.ParameterDict()
        for key,value in loaded.items():
            name=key.replace('.','__')
            if key.startswith('_actor.'):
                self.actor_weights[name]=nn.Parameter(value.clone())
            else:self.register_buffer(name,value.clone())
        for key,value in dict(default=contract['default_q'],limits=contract['joint_limits'],
            span=np.diff(contract['joint_limits'],axis=-1).ravel(),
            action_scale=.25*np.asarray(contract['training_effort'])/np.asarray(contract['kp']),
            mean=feature_mean,scale=np.maximum(feature_scale,.1)).items():
            self.register_buffer(key,torch.as_tensor(value,dtype=torch.float32).clone())
        state,priv=reference_features(neutral_motion,contract)
        self.register_buffer('neutral_state',torch.as_tensor(state[:1]).clone())
        self.register_buffer('neutral_priv',torch.as_tensor(priv[:1]).clone())
        self.goal_adapter=nn.Sequential(nn.Linear(1323,256),nn.ELU(),nn.Linear(256,256))
        nn.init.zeros_(self.goal_adapter[-1].weight);nn.init.zeros_(self.goal_adapter[-1].bias)
        self.range_adapter=nn.Linear(2048,23)
        nn.init.zeros_(self.range_adapter.weight);nn.init.zeros_(self.range_adapter.bias)

    def weight(self,key):
        name=key.replace('.','__')
        return self.actor_weights[name] if key.startswith('_actor.') else getattr(self,name)

    def linear(self,key,x):return F.linear(x,self.weight(key+'.weight'),self.weight(key+'.bias'))
    def norm(self,key,x):
        weight=self.weight(key+'.weight')
        return F.layer_norm(x,tuple(weight.shape),weight,self.weight(key+'.bias'),eps=1e-5)
    def normalize(self,key,x):
        prefix='_obs_normalizer._normalizers.'+key+'._normalizer.'
        return (x-self.weight(prefix+'running_mean'))/torch.sqrt(self.weight(prefix+'running_var')+1e-5)
    def block(self,key,x,activate=True):
        y=self.linear(key+'.mlp.1',self.norm(key+'.mlp.0',x))
        return F.mish(y) if activate else y

    def goal(self,features):
        n=features.shape[0]
        # First received reference slot: local root position starts at56+46,
        # local root rotation at56+49. All velocities use measured past.
        relative=features[:,102:105]
        delta=4*relative-features[:,52:55]
        delta=delta*delta.new_tensor([1.,1.,0.])
        delta=delta*(.6/delta.norm(dim=-1).clamp_min(1e-8)).clamp(max=1)[:,None]
        yaw=torch.atan2(features[:,107],features[:,105])
        omega=torch.stack((torch.zeros_like(yaw),torch.zeros_like(yaw),(4*yaw).clamp(-.8,.8)),-1)
        state=self.neutral_state.expand(n,-1)
        priv=self.neutral_priv.expand(n,-1)
        positions=torch.cat((torch.zeros_like(omega)[:,None],priv[:,1:73].reshape(-1,24,3)),1)
        rotation_velocity=torch.cross(omega[:,None].expand_as(positions),positions,dim=-1)
        corrected=torch.cat((priv[:,:223],priv[:,223:298]+(delta[:,None]+rotation_velocity).flatten(1),
            priv[:,298:373]+omega[:,None].expand(-1,25,-1).flatten(1)),-1)
        state=torch.cat((state[:,:-3],state[:,-3:]+omega),-1)
        encoded=torch.cat((self.normalize('state',state),self.normalize('privileged_state',corrected)),-1)
        encoded=self.linear('_backward_map.net.0',encoded)
        encoded=torch.tanh(self.norm('_backward_map.net.1',encoded))
        encoded=self.linear('_backward_map.net.3',encoded)
        neutral=16*F.normalize(encoded,dim=-1)
        return 16*F.normalize(neutral+self.goal_adapter((features-self.mean)/self.scale),dim=-1)

    def forward(self,features):
        state=torch.cat((features[:,:46],features[:,49:52],features[:,46:49]*.25),-1)
        x=torch.cat((self.normalize('state',state),self.normalize('last_action',features[:,1000:1023]),
            self.normalize('history_actor',features[:,1023:1323])),-1)
        z=self.goal(features)
        zs=self.block('_actor.embed_z.1',self.block('_actor.embed_z.0',torch.cat((x,z),-1)))
        ss=self.block('_actor.embed_s.1',self.block('_actor.embed_s.0',x))
        hidden=torch.cat((ss,zs),-1)
        for i in range(4):hidden=hidden+self.block(f'_actor.policy.{i}',hidden)
        baseline=self.default+5*self.action_scale*torch.tanh(self.block('_actor.policy.4',hidden,False))
        # The pretrained residual stream has a much larger scale than its
        # normalized output head. Give the new range head the same protection.
        range_features=F.layer_norm(hidden,(2048,)) if self.normalize_range_adapter else hidden
        if self.smooth_action_tau:
            # Opt-in candidate: residual acts before saturation, so a native
            # boundary never severs its gradient. Default/checkpoint semantics
            # below remain unchanged. New exports must record this option.
            raw=(baseline-self.limits[:,0])/self.span+self.range_adapter(range_features)
            fraction=smooth_native_fraction(raw,self.smooth_action_tau)
            target=self.limits[:,0]+self.span*fraction
            return (target-self.default)/self.span
        # A logistic change of coordinates preserves baseline targets to1e-6
        # of the native span, and permits any interior native target as the
        # adapter changes. Gradients are ordinary derivatives, no STE.
        fraction=((baseline-self.limits[:,0])/self.span).clamp(1e-6,1-1e-6)
        logits=torch.log(fraction)-torch.log1p(-fraction)+self.range_adapter(range_features)
        target=self.limits[:,0]+self.span*torch.sigmoid(logits)
        return (target-self.default)/self.span

    def target(self,action):
        return (self.default+self.span*action).clamp(self.limits[:,0],self.limits[:,1])
