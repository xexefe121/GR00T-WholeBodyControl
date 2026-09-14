from types import SimpleNamespace
import copy
import numpy as np
import pytest
from gear_sonic.utils.g1_true23_sustained_braking import SustainedBrakingFilter


class Prediction:
    joint=19
    library_sha256='fixture'
    contract=dict(joint_limits=np.tile([-2.,2.],(23,1)),kp=np.ones(23),kd=np.ones(23),native_effort=np.ones(23))
    nominal_ok=False
    def target(self,q,v,goal):return .5
    def evaluate(self,q,v,previous,first,nominal,goal):
        self.seen_previous=previous.copy()
        passed=bool(self.nominal_ok or first[19]>.1)
        return dict(passed=passed,passing_pairs=49 if passed else 0,rows=np.zeros((49,11)))


class Guard:
    adjustment=0.
    def apply(self,q,v,request,previous):
        result=request.copy();result[19]+=self.adjustment
        return result,dict(predicted_limits_satisfied=True)


def make():
    predictor=Prediction();f=SustainedBrakingFilter(predictor);f.commit_applied(np.full(23,.2))
    q=np.zeros(30);q[2]=1;q[3]=1
    return f,q,np.zeros(29),np.zeros(23),Guard()


def test_intervenes_before_guard_rejects_and_uses_actual_history():
    f,q,v,nominal,g=make();target,status=f.apply(q,v,nominal,g)
    assert status['sustained_braking_accepted'] and f.mode=='BRAKE'
    assert target[19]==.5
    np.testing.assert_array_equal(f.predictor.seen_previous,np.full(23,.2))
    np.testing.assert_array_equal(f.applied_target,np.full(23,.2))
    np.testing.assert_array_equal(np.delete(target,19),np.delete(nominal,19))


def test_guard_adjustment_is_retested_before_accepting():
    f,q,v,nominal,g=make();g.adjustment=-.5
    _,status=f.apply(q,v,nominal,g)
    assert not status['predicted_limits_satisfied']
    assert f.last_diagnostic['predictions'][-1]['label']=='actual_final_after_guard'
    assert not f.last_diagnostic['predictions'][-1]['passed']


def test_hysteresis_checked_release_and_immediate_return_to_brake():
    f,q,v,nominal,g=make();target,_=f.apply(q,v,nominal,g);f.commit_applied(target)
    f.predictor.nominal_ok=True
    for count in (1,2):
        target,_=f.apply(q,v,nominal,g);f.commit_applied(target)
        assert f.mode=='BRAKE' and f.release_counter==count
    target,_=f.apply(q,v,nominal,g);f.commit_applied(target)
    assert f.mode=='RELEASE' and abs(target[19]-.35)<1e-9
    assert any(r['label']=='checked_release' for r in f.last_diagnostic['predictions'])
    f.predictor.nominal_ok=False
    f.apply(q,v,nominal,g)
    assert f.mode=='BRAKE' and f.release_counter==0


def test_snapshot_owns_filter_memory_and_restores_decision_independent_of_labels():
    f,q,v,nominal,g=make();f.apply(q,v,nominal,g);state=f.snapshot()
    def decision(label,offset):
        # Labels and replay offsets are evaluator metadata; absent from filter API.
        f.restore(state)
        target,status=f.apply(q,v,nominal,g)
        return target,status,f.snapshot()
    a=decision('recording A',0);b=decision('unrelated recording',987654.)
    np.testing.assert_array_equal(a[0],b[0]);assert a[1]==b[1]
    for key in a[2]:
        if isinstance(a[2][key],np.ndarray):np.testing.assert_array_equal(a[2][key],b[2][key])
        else:assert a[2][key]==b[2][key]
    f.applied_target[:]=99
    np.testing.assert_array_equal(state['applied_target'],np.full(23,.2))


def test_missing_actual_history_fails_visibly():
    f,q,v,nominal,g=make();f.applied_target=None
    with pytest.raises(ValueError,match='actual applied history'):f.apply(q,v,nominal,g)
