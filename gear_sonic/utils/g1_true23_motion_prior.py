"""Training-only adversarial prior over actual native23 physical transitions.

AMP equations 6-8: https://xbpeng.github.io/projects/AMP/AMP_2021.pdf
No reference, clip identity, future pose or discriminator enters the actor.
"""
from pathlib import Path
import numpy as np
import torch
from torch import nn

STATE_WIDTH = 71
TRANSITION_WIDTH = 2 * STATE_WIDTH


def physical_features(q, v, feet, tasks):
    """Heading-invariant physical state; MuJoCo root angular qvel is local."""
    w, x, y, z = q[:, 3:7].unbind(-1)
    yaw = torch.atan2(2 * (x*y + w*z), 1 - 2 * (y*y + z*z))
    c, s = torch.cos(yaw), torch.sin(yaw)
    zero, one = torch.zeros_like(c), torch.ones_like(c)
    heading = torch.stack((c, -s, zero, s, c, zero, zero, zero, one), -1).reshape(-1, 3, 3)
    gravity = -torch.stack((2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y)), -1)
    def local(a):
        return torch.einsum('n...i,nij->n...j', a, heading)
    return torch.cat((q[:, 2:3], gravity, local(v[:, :3]), v[:, 3:6],
                      q[:, 7:], v[:, 6:], local(feet-q[:, None, :3]).flatten(1),
                      local(tasks-q[:, None, :3]).flatten(1)), -1)


class MotionPrior(nn.Module):
    def __init__(self, dataset, device, replay_capacity=50000):
        super().__init__()
        with np.load(Path(dataset), allow_pickle=False) as z:
            pairs = torch.tensor(z['transitions'], device=device)
            clip_ids = torch.tensor(z['clip_ids'], device=device)
            self.clip_names = z['clip_names'].tolist()
            if 'walk008' in self.clip_names or float(z['control_dt']) != .02:
                raise ValueError('AMP requires training clips at the native20ms control interval')
        if pairs.ndim != 2 or pairs.shape[1] != TRANSITION_WIDTH or not torch.isfinite(pairs).all():
            raise ValueError('Invalid physical transition dataset')
        if sorted(clip_ids.unique().tolist()) != list(range(len(self.clip_names))):
            raise ValueError('Every training clip needs physical transitions')
        # Equal recording weights keep long Pico from dominating the prior.
        self.pools = [pairs[clip_ids == i] for i in range(len(self.clip_names))]
        mean = torch.stack([p.mean(0) for p in self.pools]).mean(0)
        var = torch.stack([(p-mean).square().mean(0) for p in self.pools]).mean(0)
        self.register_buffer('mean', mean)
        self.register_buffer('scale', var.sqrt().clamp_min(.1))
        self.net = nn.Sequential(nn.Linear(TRANSITION_WIDTH, 256), nn.ELU(),
                                 nn.Linear(256, 128), nn.ELU(), nn.Linear(128, 1)).to(device)
        nn.init.uniform_(self.net[-1].weight, -.01, .01)
        nn.init.zeros_(self.net[-1].bias)
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=1e-4)
        self.replay = torch.empty((replay_capacity, TRANSITION_WIDTH), device=device)
        self.replay_size = self.replay_cursor = 0

    def normalized(self, pairs):
        return ((pairs-self.mean)/self.scale).clamp(-10, 10)

    @torch.no_grad()
    def reward(self, pairs, failed):
        score = self.net(self.normalized(torch.nan_to_num(pairs))).squeeze(-1)
        reward = (1-.25*(score-1).square()).clamp(0, 1)
        valid = torch.isfinite(pairs).all(-1) & ~failed
        return torch.where(valid, reward, torch.zeros_like(reward))

    def sample_expert(self, count):
        pieces = []
        for i, pool in enumerate(self.pools):
            n = count // len(self.pools) + (i < count % len(self.pools))
            pieces.append(pool[torch.randint(len(pool), (n,), device=pool.device)])
        return torch.cat(pieces)

    @torch.no_grad()
    def remember(self, pairs):
        pairs = pairs[torch.isfinite(pairs).all(-1)]
        capacity = len(self.replay)
        if len(pairs) > capacity:
            pairs = pairs[torch.randperm(len(pairs), device=pairs.device)[:capacity]]
        n = len(pairs)
        first = min(n, capacity-self.replay_cursor)
        self.replay[self.replay_cursor:self.replay_cursor+first] = pairs[:first]
        self.replay[:n-first] = pairs[first:]
        self.replay_cursor = (self.replay_cursor+n) % capacity
        self.replay_size = min(capacity, self.replay_size+n)

    def update(self, pairs, steps=8, batch_size=512):
        current = pairs[torch.isfinite(pairs).all(-1)].detach()
        if not len(current):
            raise ValueError('No finite policy transitions for the motion prior')
        stats = []
        for _ in range(steps):
            negative = current[torch.randint(len(current), (batch_size,), device=current.device)].clone()
            if self.replay_size:
                ids = torch.randint(self.replay_size, (batch_size//2,), device=current.device)
                negative[:batch_size//2] = self.replay[ids]
            positive = self.normalized(self.sample_expert(batch_size)).detach().requires_grad_(True)
            real = self.net(positive).squeeze(-1)
            fake = self.net(self.normalized(negative)).squeeze(-1)
            gradient = torch.autograd.grad(real.sum(), positive, create_graph=True)[0]
            penalty = gradient.square().sum(-1).mean()
            loss = (real-1).square().mean()+(fake+1).square().mean()+5*penalty
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(self.net.parameters(), 10)
            self.optimizer.step()
            stats.append(torch.stack((loss.detach(), real.mean().detach(), fake.mean().detach(), penalty.detach())))
        self.remember(current)
        values = torch.stack(stats).mean(0).tolist()
        return dict(zip(('loss', 'expert_score', 'policy_score', 'gradient_penalty'), values))
