"""Owned native23 controller snapshots; no inference engines or live plant pointers.

Boundary: packet t admitted, before observation/command t advances memory.
Logical timestamps retain their original epoch. A training segment's age must
not replace these timestamps or the actual command-history validity flag.
"""
from dataclasses import dataclass
import copy
import hashlib
import json
import numpy as np

SNAPSHOT_VERSION=1
BOUNDARY='packet_admitted_before_control_observation'
CONTROLLER_FIELDS=('velocity','has_applied_target','previous_native','last_reference','last_observation_time',
    'standing_captured','native_motion_seen','native_standing_captured','standing_stationary_since')
POLICY_FIELDS=('history','previous','initialized','phase','walking','last_command')
RECEIVER_FIELDS=('samples','tasks','reference_terms','cached_reference','mode','stop_last_time')
BODY_FIELDS=('alpha','epoch','first','standing_anchor')


def reference_configuration(controller):
    return dict(standing_qpos=np.asarray(controller.receiver.standing_qpos,dtype=float).tolist(),
        tasks=controller.receiver.task_spec,packet_period_s=.02,
        stale_seconds=controller.receiver.gate.stale_seconds,stop_transition_controls=100)


def wrapper_contract(controller):
    if controller.policy.phase_sync is not None:
        raise ValueError('Snapshot format1 does not support the experimental received-gait servo')
    digest=hashlib.sha256()
    for key in ('kp','kd','native_effort','joint_limits','default_q'):
        digest.update(np.asarray(controller.receiver.contract[key],dtype=np.float64).tobytes())
    model=controller.feature_model
    for key in ('body_mass','body_inertia','body_pos','body_quat','jnt_range','geom_friction','geom_size','geom_pos'):
        digest.update(np.asarray(getattr(model,key)).tobytes())
    digest.update(model.names)
    guard=controller.preview_guard
    legacy_digest=digest.hexdigest()
    for key in ('native_velocity','training_effort'):
        digest.update(np.asarray(controller.receiver.contract[key],dtype=np.float64).tobytes())
    import mujoco
    compiled=np.empty(mujoco.mj_sizeModel(model),dtype=np.uint8)
    mujoco.mj_saveModel(model,buffer=compiled)
    return dict(wrapper_version=2,snapshot_version=SNAPSHOT_VERSION,boundary=BOUNDARY,native_contract_sha256=legacy_digest,
        full_native_contract_sha256=digest.hexdigest(),compiled_model_sha256=hashlib.sha256(compiled.tobytes()).hexdigest(),
        factory_configuration_sha256=hashlib.sha256(json.dumps(controller.policy.cfg,sort_keys=True,allow_nan=False).encode()).hexdigest(),
        factory_profile=controller.policy.profile,root_velocity_feedback=controller.root_velocity_feedback,
        velocity_command_limits=controller.velocity_limits.tolist(),received_gait_servo=False,
        physics_dt=float(model.opt.timestep),target_margin_rad=controller.target_margin,
        preview=None if guard is None else dict(steps=guard.steps,delay=guard.delay,iterations=guard.iterations,reserve=guard.reserve),
        standing_capture=controller.native_standing_capture,fault_standing_capture=controller.fault_standing_capture,
        factory_prior_filter_alpha=.9,future_reference_frames=0)


def take_fields(owner,names):
    return {key:copy.deepcopy(getattr(owner,key)) for key in names if hasattr(owner,key)}


def put_fields(owner,names,values):
    for key in names:
        if key in values:setattr(owner,key,copy.deepcopy(values[key]))
        elif hasattr(owner,key):delattr(owner,key)


@dataclass(frozen=True)
class ControllerSnapshot:
    version:int
    boundary:str
    logical_time:float
    wrapper:dict
    state:dict


def snapshot_controller(controller,now):
    if not np.isfinite(now):raise ValueError('Snapshot time must be finite')
    if controller.last_observation_time is not None and now<=controller.last_observation_time:
        raise ValueError('Snapshot boundary must precede advancement of the current observation')
    r=controller.receiver
    # The receiver owns its admitted data. The gate queue is normally empty,
    # but preserving it also preserves pending packets in diagnostic snapshots.
    gate=take_fields(r.gate,tuple(k for k in vars(r.gate) if k!='contract'))
    stop=None if r.stop is None else take_fields(r.stop,tuple(k for k in vars(r.stop) if k not in ('model','data','tasks')))
    state=dict(controller=take_fields(controller,CONTROLLER_FIELDS),policy=take_fields(controller.policy,POLICY_FIELDS),
        reference_configuration=copy.deepcopy(reference_configuration(controller)),
        receiver=take_fields(r,RECEIVER_FIELDS),gate=gate,stop=stop,
        history=dict(prior=r.history.prior.copy(),terms=[x.copy() for x in r.history.terms]),
        body=take_fields(controller.body_goal,BODY_FIELDS),
        noise=None if controller.noise_rng is None else copy.deepcopy(controller.noise_rng.bit_generator.state))
    return ControllerSnapshot(SNAPSHOT_VERSION,BOUNDARY,float(now),wrapper_contract(controller),state)


def restore_controller(controller,snapshot):
    if not isinstance(snapshot,ControllerSnapshot) or snapshot.version!=SNAPSHOT_VERSION or snapshot.boundary!=BOUNDARY:
        raise ValueError('Unsupported native23 controller snapshot')
    if snapshot.wrapper!=wrapper_contract(controller):raise ValueError('Controller snapshot wrapper mismatch')
    state=copy.deepcopy(snapshot.state);r=controller.receiver
    if state.get('reference_configuration')!=reference_configuration(controller):
        raise ValueError('Controller snapshot standing/task configuration mismatch')
    if len(state['history']['terms'])!=len(r.history.terms):raise ValueError('Snapshot history layout mismatch')
    for actual,expected in zip(state['history']['terms'],r.history.terms):
        if actual.shape!=expected.shape or not np.isfinite(actual).all():raise ValueError('Invalid snapshot history')
    put_fields(controller,CONTROLLER_FIELDS,state['controller'])
    put_fields(controller.policy,POLICY_FIELDS,state['policy'])
    put_fields(r,RECEIVER_FIELDS,state['receiver'])
    for key,value in state['gate'].items():setattr(r.gate,key,value)
    r.history.prior=state['history']['prior'];r.history.terms=state['history']['terms']
    put_fields(controller.body_goal,BODY_FIELDS,state['body'])
    r.stop=None
    if state['stop'] is not None:
        import mujoco
        from gear_sonic.utils.g1_true23_causal_stop import StandingReference
        r.stop=StandingReference.__new__(StandingReference)
        r.stop.model=r.model;r.stop.data=mujoco.MjData(r.model);r.stop.tasks=r.task_spec
        for key,value in state['stop'].items():setattr(r.stop,key,value)
    if state['noise'] is not None:
        if controller.noise_rng is None:raise ValueError('Snapshot requires stochastic controller')
        controller.noise_rng.bit_generator.state=state['noise']
    return snapshot.logical_time
