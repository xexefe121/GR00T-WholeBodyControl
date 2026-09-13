"""Synthetic-testable width expansion. This module never opens a checkpoint."""
import copy
import math
import numpy as np
import torch
from torch.nn import functional as F

NAMES = ('0.weight', '0.bias', '2.weight', '2.bias', '4.weight', '4.bias')
OLD_SHAPES = ((256,1323), (256,), (256,256), (256,), (23,256), (23,))
NEW_SHAPES = ((512,1323), (512,), (512,512), (512,), (23,512), (23,))
SLICES = ((slice(0,256),slice(None)), (slice(0,256),),
          (slice(0,256),slice(0,256)), (slice(0,256),),
          (slice(None),slice(0,256)), (slice(None),))
SEED = 20260912


def _tensor(value, shape, label):
    if not isinstance(value, torch.Tensor) or value.device.type != 'cpu':
        raise ValueError(label + ': CPU tensor required')
    if value.dtype != torch.float32 or tuple(value.shape) != shape or not bool(torch.isfinite(value).all()):
        raise ValueError(label + ': finite float32 shape required')


def validate_source(saved):
    for key, expected in [('kind','direct_absolute_native23_target'), ('condition','causal'),
                          ('ordinary_final_step',71000), ('optimizer_step',6000),
                          ('context_blinded',False), ('full_state_coefficient',1.8188207859141674),
                          ('context_order','previous_action23_then_incoming_history300')]:
        if saved.get(key) != expected:
            raise ValueError('source metadata: ' + key)
    actor = saved['actor_state']
    if tuple(actor) != NAMES:
        raise ValueError('ordered six actor tensors required')
    for name, shape in zip(NAMES, OLD_SHAPES):
        _tensor(actor[name], shape, name)
    for key in ('feature_mean', 'feature_std'):
        _tensor(saved[key], (1323,), key)
    if not bool((saved['feature_std'] > 0).all()):
        raise ValueError('positive normalization std required')
    optimizer = saved['optimizer_state']
    if len(optimizer['param_groups']) != 1:
        raise ValueError('one AdamW parameter group required')
    group = optimizer['param_groups'][0]
    ids = group['params']
    if len(ids) != 6 or len(set(ids)) != 6 or set(ids) != set(optimizer['state']):
        raise ValueError('six unique optimizer identities required')
    expected = dict(lr=1e-6, weight_decay=1e-5, betas=(.9,.999), eps=1e-8,
                    foreach=False, fused=False, amsgrad=False, maximize=False,
                    capturable=False, differentiable=False, decoupled_weight_decay=True)
    for key, value in expected.items():
        if group.get(key) != value:
            raise ValueError('AdamW configuration: ' + key)
    for identifier, shape in zip(ids, OLD_SHAPES):
        state = optimizer['state'][identifier]
        if set(state) != {'step','exp_avg','exp_avg_sq'}:
            raise ValueError('three AdamW state fields required')
        _tensor(state['step'], (), 'step')
        if float(state['step']) != 6000:
            raise ValueError('all warm steps must be6000')
        for key in ('exp_avg', 'exp_avg_sq'):
            _tensor(state[key], shape, key)
        if not bool((state['exp_avg_sq'] >= 0).all()):
            raise ValueError('negative squared moment')
    if not isinstance(saved.get('rng'), dict) or not saved['rng']:
        raise ValueError('source RNG payload required')


def _expand_zero(old, shape, block):
    value = torch.zeros(shape, dtype=torch.float32)
    value[block] = old.detach().clone()
    return value


def expand_source(saved, seed=SEED):
    """Return an initialization capsule; preserve source tensors and global RNG."""
    validate_source(saved)
    if type(seed) is not int or seed != SEED:
        raise ValueError('fixed local expansion seed required')
    generator = torch.Generator(device='cpu').manual_seed(seed)
    actor = {name:_expand_zero(saved['actor_state'][name], shape, block)
             for name, shape, block in zip(NAMES, NEW_SHAPES, SLICES)}
    # Only randomize incoming parameters of added neurons. Old-to-output paths
    # keep all old values, and both new-to-old/output paths start exactly zero.
    with torch.no_grad():
        for name, block, fan_in in [('0.weight',(slice(256,None),slice(None)),1323),
                                    ('0.bias',(slice(256,None),),1323),
                                    ('2.weight',(slice(256,None),slice(None)),512),
                                    ('2.bias',(slice(256,None),),512)]:
            bound = 1 / math.sqrt(fan_in)
            actor[name][block].uniform_(-bound, bound, generator=generator)
    optimizer = copy.deepcopy(saved['optimizer_state'])
    for identifier, shape, block in zip(optimizer['param_groups'][0]['params'], NEW_SHAPES, SLICES):
        for key in ('exp_avg', 'exp_avg_sq'):
            optimizer['state'][identifier][key] = _expand_zero(
                saved['optimizer_state']['state'][identifier][key], shape, block)
    return dict(kind='unreleased_width512_initialization', preparation_only=True,
                actor_state=actor, optimizer_state=optimizer,
                feature_mean=saved['feature_mean'].detach().clone(),
                feature_std=saved['feature_std'].detach().clone(),
                rng_to_restore=copy.deepcopy(saved['rng']),
                expansion_seed=seed, expansion_generator_state=generator.get_state().clone(),
                source_ordinary_step=71000, optimizer_start_step=6000,
                old_width=256, new_width=512, parameter_count=6)


class _ZeroPreservingAdd(torch.autograd.Function):
    @staticmethod
    def forward(ctx, original, added):
        if original.shape != added.shape or original.dtype != added.dtype:
            raise ValueError('equal-shaped same-dtype additions required')
        return torch.where(added == 0, original, original + added)

    @staticmethod
    def backward(ctx, gradient):
        # Exact derivative of real addition, including at added==0. This keeps
        # newly zero outgoing blocks trainable while preserving signed zeros.
        return gradient, gradient


def zero_preserving_add(original, added):
    return _ZeroPreservingAdd.apply(original, added)


class WiderContextTarget(torch.nn.Module):
    def __init__(self, mean, std):
        super().__init__()
        if mean.shape != (1323,) or std.shape != (1323,) or mean.dtype != np.float32 or std.dtype != np.float32:
            raise ValueError('1323 float32 normalization required')
        if not np.isfinite(mean).all() or not np.isfinite(std).all() or np.any(std <= 0):
            raise ValueError('finite positive normalization required')
        self.register_buffer('feature_mean',torch.from_numpy(mean.copy()))
        self.register_buffer('feature_std',torch.from_numpy(std.copy()))
        # Default Linear initialization is overwritten by the capsule. Avoid
        # disturbing the source CPU RNG; this constructor never touches CUDA.
        with torch.random.fork_rng(devices=[]):
            self.actor = torch.nn.Sequential(torch.nn.Linear(1323,512),torch.nn.ELU(),
                torch.nn.Linear(512,512),torch.nn.ELU(),torch.nn.Linear(512,23))

    def forward(self, features):
        if features.dtype != torch.float32 or features.ndim != 2 or features.shape[1] != 1323:
            raise ValueError('1323 public float32 features required')
        normalized = (features-self.feature_mean)/self.feature_std
        first, second, last = self.actor[0], self.actor[2], self.actor[4]
        old_first = F.linear(normalized[:,:1000].contiguous(),first.weight[:256,:1000].contiguous(),first.bias[:256])
        old_context = F.linear(normalized[:,1000:].contiguous(),first.weight[:256,1000:].contiguous(),None)
        old_hidden = F.elu(old_first+old_context)
        new_hidden = F.elu(F.linear(normalized,first.weight[256:,:].contiguous(),first.bias[256:]))
        old_second = F.linear(old_hidden,second.weight[:256,:256].contiguous(),second.bias[:256])
        added_second = F.linear(new_hidden,second.weight[:256,256:].contiguous(),None)
        old_hidden2 = F.elu(zero_preserving_add(old_second,added_second))
        new_hidden2 = F.elu(F.linear(torch.cat((old_hidden,new_hidden),dim=1),
                                    second.weight[256:,:].contiguous(),second.bias[256:]))
        original = F.linear(old_hidden2,last.weight[:,:256].contiguous(),last.bias)
        added = F.linear(new_hidden2,last.weight[:,256:].contiguous(),None)
        return zero_preserving_add(original,added)


def proposed_rate(index):
    if type(index) is not int or not 0 <= index < 10000:
        raise ValueError('fixed10000-update index required')
    if index < 250:
        return 1e-6 + (1e-5-1e-6) * index / 249
    return 1e-6 + .5 * (1e-5-1e-6) * (1+math.cos(math.pi*(index-250)/9749))
