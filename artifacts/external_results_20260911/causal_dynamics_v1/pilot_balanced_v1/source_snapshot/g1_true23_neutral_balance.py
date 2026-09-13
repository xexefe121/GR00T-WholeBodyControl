"""Batched frozen native23 neutral balance, shared by training and runtime."""
import torch
import torch.nn.functional as F
from gear_sonic.utils.g1_true23_bfmzero_inference import BFMZeroInference,reference_features
from gear_sonic.utils.g1_true23_bfm_residual import bfm_state_and_terms,heading_matrix

class NeutralBalance:
    def __init__(self,weights,contract,neutral_motion,device='cpu'):
        self.policy=BFMZeroInference(weights)
        self.policy.weights={k:v.to(device) for k,v in self.policy.weights.items()}
        state,priv=reference_features(neutral_motion,contract)
        self.state=torch.tensor(state[:1],device=device)
        self.priv=torch.tensor(priv[:1],device=device)
        self.default=torch.tensor(contract['default_q'],dtype=torch.float32,device=device)
        self.scale=.25*torch.tensor(contract['training_effort'],dtype=torch.float32,device=device)/torch.tensor(contract['kp'],dtype=torch.float32,device=device)
        self.limits=torch.tensor(contract['joint_limits'],dtype=torch.float32,device=device)

    @torch.inference_mode()
    def target(self,q,v,prior,history,anchor):
        n=q.shape[0];h,yaw=heading_matrix(q[:,3:7])
        delta=torch.zeros(n,3,device=q.device);delta[:,:2]=4*(anchor[:,:2]-q[:,:2])-v[:,:2]
        delta*=(.6/delta.norm(dim=-1).clamp_min(1e-8)).clamp(max=1)[:,None]
        dy=anchor[:,2]-yaw;omega=torch.zeros_like(delta)
        omega[:,2]=(4*torch.atan2(torch.sin(dy),torch.cos(dy))).clamp(-.8,.8)
        state=self.state.expand(n,-1).clone();priv=self.priv.expand(n,-1).clone()
        positions=torch.cat((torch.zeros(n,1,3,device=q.device),priv[:,1:73].reshape(n,24,3)),1)
        local=torch.einsum('nji,nj->ni',h,delta)
        priv[:,223:298]+=(local[:,None]+torch.cross(omega[:,None].expand_as(positions),positions,dim=-1)).flatten(1)
        priv[:,298:373]+=omega[:,None].expand(-1,25,-1).flatten(1);state[:,-3:]+=omega
        goal=16*F.normalize(self.policy.backward(state,priv),dim=-1)
        sensed,_=bfm_state_and_terms(q,v,prior,self.default)
        raw=5*self.policy.actor(sensed,prior,history,goal)
        return (self.default+raw*self.scale).clamp(self.limits[:,0],self.limits[:,1])


