import numpy as np
import pytest
from audit_math import balanced_moments, phase_indices, response_error, targets


def test_between_group_variance_and_equal_population():
    x = np.array([[0.],[10.],[10.],[10.]],np.float32)
    mean,variance,mean32,std32 = balanced_moments(x,[np.array([0]),np.array([1,2,3])])
    np.testing.assert_allclose(mean,[5.],rtol=0,atol=1e-14)
    np.testing.assert_allclose(variance,[25.],rtol=0,atol=1e-14)
    np.testing.assert_array_equal(mean32,[5.]);np.testing.assert_array_equal(std32,[5.])


def test_zero_variance_floor():
    x = np.zeros((4,3),np.float32)
    _,_,_,std = balanced_moments(x,[np.arange(4)])
    np.testing.assert_array_equal(std,np.full(3,.05,np.float32))


@pytest.mark.parametrize('groups',[[np.array([0]),np.array([0,1,2])],[np.array([0,1])],[np.array([],dtype=int),np.arange(3)],[np.array([0,0,1,2])]])
def test_bad_partition_rejected(groups):
    with pytest.raises(ValueError):balanced_moments(np.zeros((3,2),np.float32),groups)


def test_response_promotes_before_subtraction():
    # f32 subtraction would lose the .25 difference against this magnitude.
    endpoint=np.array([[2**24]],np.float32);nominal=np.array([[.25]],np.float32)
    actual=response_error(endpoint,nominal,endpoint.astype(np.float64)-.25,np.zeros((1,1),np.float64),np.ones(1,np.float32))
    np.testing.assert_array_equal(actual,np.zeros((1,1)))


def test_runtime_float64_clamp_and_unconstrained_output():
    output=np.full((2,23),2,np.float32);output[1]=-2
    default=np.zeros(23,np.float64);span=np.ones(23,np.float32)
    limits=np.tile(np.array([-.087267,.2618],np.float64),(23,1))
    raw,applied,counts=targets(output,default,span,limits)
    np.testing.assert_array_equal(raw,output.astype(np.float64))
    np.testing.assert_array_equal(applied[0],limits[:,1]);np.testing.assert_array_equal(applied[1],limits[:,0])
    assert counts=={'clipped_rows':2,'clipped_components':46}


def test_phase_successor_partitions():
    nominal=phase_indices();physical=phase_indices(((99,819,100),)*3)
    assert sum(map(len,nominal))==9904 and sum(map(len,physical))==3054
    assert list(nominal[6][:2])==[2038,2039]
    assert [len(x) for x in physical]==[99,819,100]*3
