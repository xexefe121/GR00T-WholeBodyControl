import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from gear_sonic.scripts.evaluate_g1_true23_causal_dynamics import load_case
from gear_sonic.utils.g1_true23_sensor_feedback import Native23SensorFeedback,SensorEffects


@pytest.fixture(scope='module')
def model_contract():
    model,contract,*_=load_case('walk002')
    return model,contract


def sample(contract,t=0.):
    quat=Rotation.from_euler('z',.4).as_quat()[[3,0,1,2]]
    return np.r_[t,contract['default_q'],np.zeros(23),quat,np.zeros(3),[0.,0.,9.81]]


def test_stationary_sensor_feedback_has_one_alignment_and_no_root_inputs(model_contract):
    model,c=model_contract;f=Native23SensorFeedback(model)
    q,v,status=f.initialize(sample(c),supported_stationary=True)
    np.testing.assert_allclose(q[3:7],[1.,0.,0.,0.],atol=1e-12)
    for i in range(1,101):q,v,status=f.update(sample(c,i*.02),now=i*.02+.002)
    np.testing.assert_allclose(q[:2],0.,atol=1e-12)
    np.testing.assert_allclose(v,0.,atol=1e-12)
    assert q.shape==(30,) and v.shape==(29,)
    assert status['ground_truth_fallback'] is False
    assert f.contract()['measured_root_position_input'] is False


@pytest.mark.parametrize('kind',['duplicate','stale','nan','bad_quaternion'])
def test_bad_sensor_sample_latches_fault(model_contract,kind):
    model,c=model_contract;f=Native23SensorFeedback(model);f.initialize(sample(c),supported_stationary=True)
    value=sample(c,.02);now=.022
    if kind=='duplicate':value[0]=0.
    if kind=='stale':now=.08
    if kind=='nan':value[54]=np.nan
    if kind=='bad_quaternion':value[47:51]=0.
    with pytest.raises(ValueError):f.update(value,now=now)
    with pytest.raises(ValueError,match='latched'):f.update(sample(c,.04),now=.042)


def test_delay_uses_actual_priming_samples_and_deterministic_noise(model_contract):
    model,c=model_contract;e=SensorEffects(delay_controls=2,gyro_std_rad_s=.0001,seed=12)
    a,b=Native23SensorFeedback(model,effects=e),Native23SensorFeedback(model,effects=e)
    for f in (a,b):
        f.initialize(sample(c),supported_stationary=True)
        f.prime_delay(sample(c,.02));f.prime_delay(sample(c,.04))
    qa,va,sa=a.update(sample(c,.06),now=.062)
    qb,vb,sb=b.update(sample(c,.06),now=.062)
    np.testing.assert_array_equal(qa,qb);np.testing.assert_array_equal(va,vb)
    assert sa==sb and sa['sensor_timestamp_s']==.02
    assert abs(sa['sensor_age_s']-.042)<1e-12


def test_stationary_initialization_cannot_be_assumed_from_freefall(model_contract):
    model,c=model_contract;f=Native23SensorFeedback(model);value=sample(c);value[54:]=0.
    with pytest.raises(ValueError,match='quiet supported standing'):f.initialize(value,supported_stationary=True)
    assert f.fault is not None


def test_delay_warmup_failure_latches_until_explicit_reinitialization(model_contract):
    model,c=model_contract;f=Native23SensorFeedback(model,effects=SensorEffects(delay_controls=2))
    f.initialize(sample(c),supported_stationary=True)
    with pytest.raises(ValueError,match='warmup fault'):f.prime_delay(sample(c))
    with pytest.raises(ValueError,match='latched'):f.prime_delay(sample(c,.02))
    with pytest.raises(ValueError,match='latched'):f.update(sample(c,.04),now=.042)
