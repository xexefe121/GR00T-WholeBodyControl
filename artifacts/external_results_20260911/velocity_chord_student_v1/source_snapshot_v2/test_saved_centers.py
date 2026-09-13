"""Read-only saved-center reconstruction test; no ONNX sessions or physics imports."""
from types import SimpleNamespace
import numpy as np
from chord_common import NEW,BASE,read,local,exact,write,sha
from saved_committed_map import difference_function
from generate_velocity_chords import reconstruct_centers
from gear_sonic.utils.g1_true23_bfm_seed_observations import BFMHistory,state_and_terms

def main():
    c=read(local('Z:/codex/GR00T-WholeBodyControl-sonic-transfer-23dof/artifacts/teleop_six_hour_20260910/mjbatch_native23_inputs_v1/contract.json'))
    history=BFMHistory()
    seed=SimpleNamespace(history=history,_terms=lambda q,v,a:state_and_terms(q[7:],v[6:],q[3:7],v[3:6],a,np.asarray(c['default_q'])))
    center=reconstruct_centers(seed,c,np.asarray(c['joint_limits']),difference_function())
    assert len(center['control'])==3057
    for dataset in range(3):
        ids=center['dataset']==dataset
        np.testing.assert_array_equal(center['control'][ids],np.arange(250,1269))
    for k,v in history.data.items():exact(v,np.zeros_like(v),'test history remains unchanged')
    source=BASE/'source_draft_v1'
    report=dict(pass_all=True,nominal_rows=3057,all_nominal_targets_full_gain_difference_byteexact=True,
        all_state_history_prior_exact=True,source_sha256={p.name:sha(p) for p in source.glob('*.py')},
        model_inference_calls=0,physics_steps=0,optimizer_updates=0)
    output=BASE/'saved_center_pure_test_v2.json'
    assert not output.exists()
    write(output,report)
    print('PURE SAVED CENTER TEST PASS:3057; no inference/physics/optimizer')

if __name__=='__main__':main()
