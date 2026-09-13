"""Exact qualified H30 cost terms on a five-control native forecast prefix."""
import time
import numpy as np
from gear_sonic.utils.g1_true23_mjbatch_mpc import Native23Tracker
from gear_sonic.utils.g1_true23_mjbatch_model import position_servo_copy

COMPONENTS=('world_body_position','body_rotation_log','joint_position','generalized_velocity',
    'lower_joint_margin','upper_joint_margin','native_speed_soft_margin')
SLICES=(slice(0,18),slice(18,36),slice(36,59),slice(59,88),slice(88,111),slice(111,134),slice(134,157))


class PrefixTrackingCost:
    def __init__(self,native,contract,motion):
        self.tracker=Native23Tracker(position_servo_copy(native,np.asarray(contract['kp']),np.asarray(contract['kd']),np.asarray(contract['native_effort'])),
            contract,motion,horizon=30,threads=2,all_joint_limit_margin=.05,all_joint_limit_weight=2000,
            relative_foot_weight=0,fd_epsilon=1e-6,hard_feasibility=True)
        self.calls=0
        def forbidden(*args,**kwargs):raise RuntimeError('Runtime planner dynamics, derivatives and optimization are forbidden.')
        for name in ('rollout','advance','step','linearize','expand'):setattr(self.tracker,name,forbidden)

    def score(self,control,target,forecast,feasible):
        started=time.perf_counter();tracker=self.tracker;target=np.asarray(target,np.float64)
        if type(feasible) is not bool or type(control) is not int or control<250:raise ValueError('Invalid cost scope.')
        if target.shape!=(23,) or not np.isfinite(target).all():raise ValueError('Invalid scoring target.')
        steps=len(forecast['physics_torque'])
        if not 0<=steps<=50 or (feasible and steps!=50):raise ValueError('Cost requires the fixed100ms checked forecast.')
        if len(forecast['physics_qpos'])!=steps+1 or len(forecast['physics_qvel'])!=steps+1:raise ValueError('Forecast state count mismatch.')
        count=steps//10+1;indices=np.arange(count)*10
        states=np.concatenate((forecast['physics_qpos'][indices],forecast['physics_qvel'][indices]),axis=1)
        tracker.window(control+10);features=tracker.features(states)
        residual=tracker.residual(np.arange(count),features)
        if residual.shape!=(count,157) or not np.isfinite(residual).all():raise ValueError('Nonfinite or incompatible original tracking residual.')
        squared=residual**2;state_cost=np.sum(squared,axis=1)
        components=np.stack([np.sum(squared[:,sl],axis=-1) for sl in SLICES],axis=1)
        references=tracker.target_reference(np.arange(5))
        input_cost=tracker.control_weight*np.sum((target[None]-references)**2,axis=1)
        complete=count==6
        score=float(np.sum(state_cost)+np.sum(input_cost)) if complete else None
        if score is not None and not np.isfinite(score):raise ValueError('Nonfinite complete prefix cost.')
        arrays=dict(state=states.copy(),features=features.copy(),residual=residual.copy(),component_cost=components,
            state_cost=state_cost,input_cost=input_cost,target_reference=references.copy(),
            state_goal_frame=np.minimum(control+10+np.arange(count),len(tracker.states)-1),
            input_goal_frame=np.minimum(control+11+np.arange(5),len(tracker.states)-1))
        self.calls+=1
        report=dict(available_state_knots=count,complete_five_control_cost=complete,
            feasible_for_cost_ranking=feasible and complete,initial_state_cost=float(state_cost[0]),
            five_control_prefix_state_and_input_sum=score,five_input_cost_sum=float(np.sum(input_cost)),
            component_sums={name:float(np.sum(components[:,i])) for i,name in enumerate(COMPONENTS)} if complete else None,
            cost_evaluation_ms=(time.perf_counter()-started)*1000,
            full_H30_cost_not_evaluated=True,unsafe_partial_costs_not_ranked=True)
        return report,arrays
