"""Synthetic runtime stubs only; no task model, native dynamics or actual arrays."""
import copy
import numpy as np
import pytest
from causal_features import CausalFeatures, incoming_context
from direct_runtime import actual_previous_action, exact
from test_direct_adapter import runtime, StubSession


def setup(condition='causal', head=None):
    r,q,v = runtime(head=head)
    r.seed.recorded_controls = 250
    r.seed.previous_action[:] = np.linspace(-2, 2, 23, dtype=np.float32)
    for index, value in enumerate(r.seed.history.data.values()):
        value[:] = np.arange(value.size, dtype=np.float32).reshape(value.shape) + 100*index
    original = r.features
    mean = np.linspace(-1, 1, 323, dtype=np.float32)
    r.features = CausalFeatures(original, r.seed, condition, mean)
    return r,q,v,mean


def test_context_matches_proposal_incoming_state_and_stays_uncommitted():
    r,q,v,_ = setup()
    before = incoming_context(r.seed)
    proposed = r.propose(250,q,v)
    assert exact(proposed['features'][1000:], before)
    assert exact(proposed['features'][1000:1023], proposed['previous_action'])
    assert exact(proposed['features'][1023:], proposed['history'])
    assert exact(incoming_context(r.seed), before)
    assert r.head.inner.feeds[0]['features'].shape == (1,1323)


def test_next_context_uses_applied_target_and_exactly_one_shift():
    r,q,v,_ = setup(head=StubSession(np.full((1,23),10,np.float32)))
    before = {k:a.copy() for k,a in r.seed.history.data.items()}
    first = r.propose(250,q,v)
    r.commit(first)
    second = r.propose(251,q,v)
    assert exact(second['features'][1000:1023],actual_previous_action(first['target'],r.c))
    assert exact(second['features'][1023:],second['history'])
    for key,value in r.seed.history.data.items():
        assert exact(value[1:],before[key][:3])
    assert exact(r.seed.history.data['actions'][0],first['previous_action'])


def test_changed_current_state_does_not_rewrite_lagged_context():
    r,q,v,_ = setup()
    first = r.features(q,v,261)
    moved = q.copy();moved[7] += .1
    second = r.features(moved,v,261)
    assert not exact(first[:1000],second[:1000])
    assert exact(first[1000:],second[1000:])


def test_blinded_context_normalizes_to_exact_zero_without_hiding_actual_history():
    r,q,v,mean = setup('blinded')
    proposed = r.propose(250,q,v)
    assert exact(proposed['features'][1000:],mean)
    assert np.count_nonzero(proposed['features'][1000:]-mean) == 0
    assert np.count_nonzero(proposed['features'][1000:].astype(np.float64)-mean.astype(np.float64)) == 0
    assert not exact(proposed['history'],mean[23:])


def test_exception_preserves_extended_input_without_advancing_seed():
    r,q,v,_ = setup(head=StubSession(np.zeros((1,23),np.float32),fail=True))
    before = incoming_context(r.seed)
    with pytest.raises(RuntimeError):r.propose(250,q,v)
    assert exact(incoming_context(r.seed),before)
    assert r.context['head_input_features'].shape == (1,1323)
    assert r.seed.recorded_controls == 250


def test_terminal_preserves_bfm_and_extended_trace_shape():
    r,q,v,_ = setup();r.seed.recorded_controls = 1269
    proposed = r.propose(1269,q,v)
    assert proposed['features'].shape == (1323,)
    assert not np.any(proposed['features'])
    assert r.counts['2_actor_returned'] == r.counts['2_backward_returned'] == 1
    assert r.counts['1_head_returned'] == 0


def test_signed_zero_context_bytes_owned_and_invalid_schema_rejected():
    r,q,v,_ = setup();r.seed.previous_action[0] = -0.0
    value = incoming_context(r.seed)
    assert np.signbit(value[0])
    r.seed.previous_action[0] = 1
    assert np.signbit(value[0])
    r.seed.history.data['actions'] = r.seed.history.data['actions'].astype(np.float64)
    with pytest.raises(ValueError):r.features(q,v,261)
