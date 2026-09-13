"""Independent-audit mechanics only; no graph, policy, or physics execution."""
from pathlib import Path
import numpy as np
import pytest
from audit import actual_observations, CountedSession, Proof, same
from gear_sonic.utils.g1_true23_bfm_seed_observations import BFMHistory, state_and_terms


def test_independent_history_matches_original_timing_and_signed_zeros():
    rng=np.random.default_rng(4201)
    trace=dict(qpos=np.zeros((8,30)),qvel=np.zeros((8,29)),target=rng.normal(size=(7,23)))
    trace['qpos'][:,3]=1.;trace['qpos'][1:,7:]=rng.normal(size=(7,23));trace['qvel'][1:]=rng.normal(size=(7,29))
    c=dict(default_q=np.zeros(23),kp=np.ones(23),training_effort=np.ones(23))
    history=BFMHistory();prior=np.zeros(23,np.float32)
    for control,q,dq,previous,state,flat,named in actual_observations(trace,c,6):
        expected,terms=state_and_terms(q[7:],dq[6:],q[3:7],dq[3:6],prior,c['default_q'])
        assert same(previous,prior) and same(state,expected)
        for key in named:assert same(named[key],history.data[key])
        assert same(flat,history.before_update(terms))
        prior=((trace['target'][control]-c['default_q'])*c['kp']/(.25*c['training_effort'])).astype(np.float32)


def test_graph_counter_allows_only_fixed_selected_budget():
    actual=[0]
    class Session:
        def run(self,*a,**k):actual[0]+=1;return None
    calls=dict(actor=0,backward=0)
    counted=CountedSession(Session(),'actor',calls)
    for _ in range(6847):counted.run(None,{})
    assert actual[0]==calls['actor']==6847
    with pytest.raises(RuntimeError,match='limit exceeded'):counted.run(None,{})
    assert actual[0]==6847


@pytest.mark.parametrize('actual,expected', [(np.array([-0.],np.float32),np.array([0.],np.float32)),
    (np.array([1.],np.float32),np.array([1.],np.float64)),(np.zeros(3),np.zeros(4))])
def test_first_mismatch_preserved_with_original_context(tmp_path,actual,expected):
    proof=Proof(tmp_path);proof.clip='pico';proof.control=250
    proof.context=dict(integration=np.arange(291,dtype=np.float64),history=np.zeros(300,np.float32))
    with pytest.raises(ValueError):proof.equal('exact_test',actual,expected)
    with np.load(tmp_path/'first_mismatch.npz',allow_pickle=False) as saved:
        assert same(saved['actual'],actual) and same(saved['expected'],expected)
        assert saved['control'].item()==250 and same(saved['integration'],proof.context['integration'])
    report=proof.save();assert report['all_pass'] is False and report['comparisons']==1


@pytest.mark.parametrize('name', ['student_linear_runtime.py','terminal_yaw4_goal.py',
    'gear_sonic/utils/g1_true23_mpc_student.py','gear_sonic/utils/g1_true23_mjbatch_bfm_seed.py'])
def test_feature_and_baseline_helpers_are_original_copies(name):
    here=Path(__file__).parent
    original=here.parent.parent/'bfm_entry250_labels_v1/source_snapshot_v1'
    assert (here/name).read_bytes()==(original/name).read_bytes()
