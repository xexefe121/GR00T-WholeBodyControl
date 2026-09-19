"""Opt-in simulation-only shoulder filter with bounded feedback continuation."""
import copy
import ctypes as ct
import hashlib
from pathlib import Path
import numpy as np

DOUBLE=ct.POINTER(ct.c_double)
CONFIG=dict(version=1,joint='right_shoulder_roll_joint',horizon_seconds=.08,
    delay_pairs=49,tail_delay='next delay repeated for remaining two updates',
    backup_acceleration_position_gain=400.,backup_acceleration_velocity_gain=40.,
    capture_margin_rad=.15,terminal_margin_rad=.02,terminal_speed_rad_s=.25,
    terminal_consecutive_endpoints=2,release_margin_rad=.12,release_speed_rad_s=.1,
    release_consecutive_updates=3,release_target_step_rad=.15,reserve_rad=.0001,
    candidate_uniform_points=9,simulation_only=True,formal_safety_guarantee=False)


def ptr(x):return x.ctypes.data_as(DOUBLE)


class Predictor:
    def __init__(self,model,contract,library):
        self.model=model;self.contract=contract
        joint=model.joint(CONFIG['joint']);self.joint=int(joint.qposadr[0])-7
        if int(joint.dofadr[0])-6!=self.joint:raise ValueError('Joint mapping mismatch')
        np.testing.assert_array_equal(model.jnt_range[joint.id],contract['joint_limits'][self.joint])
        self.library_sha256=hashlib.sha256(Path(library).read_bytes()).hexdigest()
        self.lib=ct.CDLL(str(library))
        self.lib.braking_create.argtypes=[ct.c_void_p,ct.c_int]+[DOUBLE]*5;self.lib.braking_create.restype=ct.c_void_p
        self.lib.braking_destroy.argtypes=[ct.c_void_p]
        self.lib.braking_target.argtypes=[ct.c_void_p,DOUBLE,DOUBLE,ct.c_double];self.lib.braking_target.restype=ct.c_double
        self.lib.braking_evaluate.argtypes=[ct.c_void_p]+[DOUBLE]*5+[ct.c_double,DOUBLE]
        arrays=[np.ascontiguousarray(contract[k],float) for k in ('kp','kd','native_effort','native_velocity','joint_limits')]
        self.address=self.lib.braking_create(model._address,self.joint,*map(ptr,arrays))
        if not self.address:raise ValueError('Predictor rejected native model')

    def target(self,q,v,goal):
        q,v=[np.ascontiguousarray(x,float) for x in (q,v)]
        return self.lib.braking_target(self.address,ptr(q),ptr(v),goal)

    def evaluate(self,q,v,previous,first,nominal,goal):
        arrays=[np.ascontiguousarray(x,float) for x in (q,v,previous,first,nominal)]
        if not all(np.isfinite(x).all() for x in arrays):raise ValueError('Nonfinite braking prediction input')
        rows=np.empty((49,11),float)
        self.lib.braking_evaluate(self.address,*map(ptr,arrays),goal,ptr(rows))
        return dict(passed=bool(np.all(rows[:,0]==1)),passing_pairs=int(rows[:,0].sum()),rows=rows)

    def __del__(self):
        if getattr(self,'address',None):self.lib.braking_destroy(self.address);self.address=None


class SustainedBrakingFilter:
    def __init__(self,predictor):
        self.predictor=predictor;self.joint=predictor.joint;self.contract=predictor.contract
        self.mode='TRACK';self.release_counter=0;self.previous_correction=None
        self.continuation=None;self.applied_target=None;self.last_diagnostic=None

    def configuration(self):return dict(CONFIG,backend_sha256=self.predictor.library_sha256)

    def snapshot(self):
        return copy.deepcopy(dict(mode=self.mode,release_counter=self.release_counter,
            previous_correction=self.previous_correction,continuation=self.continuation,applied_target=self.applied_target))

    def restore(self,state):
        if state['mode'] not in ('TRACK','BRAKE','RELEASE'):raise ValueError('Invalid braking mode')
        for key,value in copy.deepcopy(state).items():setattr(self,key,value)

    def commit_applied(self,target):
        target=np.asarray(target,float)
        if target.shape!=(23,) or not np.isfinite(target).all():raise ValueError('Invalid applied target')
        self.applied_target=target.copy()

    def apply(self,q,v,nominal,guard):
        if self.applied_target is None:raise ValueError('Braking filter requires actual applied history')
        q,v,nominal=[np.asarray(x,float) for x in (q,v,nominal)]
        if q.shape!=(30,) or v.shape!=(29,) or nominal.shape!=(23,):raise ValueError('Invalid braking observation')
        j=self.joint;lo,hi=self.contract['joint_limits'][j];previous=self.applied_target.copy()
        goal=float(np.clip(q[7+j],lo+CONFIG['capture_margin_rad'],hi-CONFIG['capture_margin_rad']))
        if self.mode!='TRACK' and self.continuation is not None:goal=self.continuation['goal_rad']
        predictions=[]
        def check(target,label):
            result=self.predictor.evaluate(q,v,previous,target,nominal,goal)
            predictions.append(dict(label=label,target_rad=float(target[j]),**result));return result
        nominal_test=check(nominal,'nominal_then_feedback_braking')
        interior=min(q[7+j]-lo,hi-q[7+j])>=CONFIG['release_margin_rad'] and abs(v[6+j])<=CONFIG['release_speed_rad_s']
        release_ready=bool(interior and nominal_test['passed'])
        before=self.mode
        if self.mode=='TRACK' and nominal_test['passed']:
            candidate=nominal.copy()
        else:
            self.release_counter=self.release_counter+1 if release_ready else 0
            if self.mode=='TRACK' or not release_ready:self.mode='BRAKE'
            if self.mode=='BRAKE' and self.release_counter>=CONFIG['release_consecutive_updates']:self.mode='RELEASE'
            candidate=None
            if self.mode=='RELEASE':
                release=nominal.copy();release[j]=previous[j]+np.clip(nominal[j]-previous[j],-CONFIG['release_target_step_rad'],CONFIG['release_target_step_rad'])
                if check(release,'checked_release')['passed']:candidate=release
                else:self.mode='BRAKE';self.release_counter=0
            if candidate is None:
                c=self.contract;angle=q[7+j];speed=v[6+j]
                values=[self.predictor.target(q,v,goal),angle,previous[j],
                    angle+(c['native_effort'][j]+c['kd'][j]*speed)/c['kp'][j],
                    angle+(-c['native_effort'][j]+c['kd'][j]*speed)/c['kp'][j],*np.linspace(lo,hi,9)]
                if self.previous_correction is not None:values.append(self.previous_correction)
                # Feedback option first, then remaining legal seeds closest to actor.
                values=list(dict.fromkeys(np.clip(values,lo,hi).tolist()))
                values=values[:1]+sorted(values[1:],key=lambda x:abs(x-nominal[j]))
                for value in values:
                    trial=nominal.copy();trial[j]=value
                    if check(trial,'intervene_then_feedback_braking')['passed']:candidate=trial;break
        established=candidate is not None
        final,status=guard.apply(q,v,nominal if candidate is None else candidate,previous=previous)
        checked_final=None
        if established:
            checked_final=check(final,'actual_final_after_guard')
            intact=np.array_equal(np.delete(final,j),np.delete(nominal,j))
            established=bool(checked_final['passed'] and intact and status['predicted_limits_satisfied'])
        if not established:
            status=dict(status,predicted_limits_satisfied=False,sustained_braking_rejected=True)
        else:
            self.previous_correction=float(final[j]);self.continuation=dict(goal_rad=goal,horizon_seconds=.08)
            if self.mode=='RELEASE' and abs(final[j]-nominal[j])<1e-9:self.mode='TRACK';self.release_counter=0
        self.last_diagnostic=dict(mode_before=before,mode=self.mode,release_counter=self.release_counter,
            release_ready=release_ready,accepted=established,nominal_passed=nominal_test['passed'],
            applied_previous_target=previous,actor_target=nominal.copy(),final_target=final.copy(),
            correction_rad=float(final[j]-nominal[j]),capture_goal_rad=goal,predictions=predictions,
            reason=None if established else 'no_acceptable_final_command_and_braking_continuation',
            configuration=self.configuration())
        return final,dict(status,sustained_braking_mode=self.mode,sustained_braking_accepted=established)
