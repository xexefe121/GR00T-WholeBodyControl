"""Pure tensor tests: no model, optimizer, BFM, or physical state."""
from pathlib import Path
import json
import hashlib
import numpy as np
import torch
from chord_training_objective import chord_loss,sample_pairs,strata_for

def main():
    torch.set_num_threads(1)
    center=torch.zeros((3057,23),requires_grad=True)
    probe=torch.ones((1152,23),requires_grad=True)
    picked=torch.zeros(576,dtype=torch.int64)
    base=torch.zeros((3057,23),dtype=torch.float64)
    pair=torch.zeros((576,2,23),dtype=torch.float64)
    loss=chord_loss(center,probe,picked,base,pair,base,pair,torch.ones(23))
    assert float(loss.detach())==1.
    loss.backward()
    torch.testing.assert_close(center.grad[0],torch.full((23,),-2/23),atol=1e-6,rtol=1e-6)
    assert torch.count_nonzero(center.grad[1:])==0
    torch.testing.assert_close(probe.grad,torch.full((1152,23),2/(1152*23)),atol=1e-9,rtol=1e-6)
    matched=chord_loss(center.detach(),probe.detach(),picked,base,pair,base,torch.ones_like(pair),torch.ones(23))
    assert float(matched)==0.
    # Distinct cells/signs/output axes expose incorrect reductions or weighting.
    varied=torch.empty((576,2,23),dtype=torch.float32)
    for cell in range(9):
        for sign in range(2):
            varied[cell*64:(cell+1)*64,sign]=torch.arange(1,24,dtype=torch.float32)*(cell+1)*(sign+1)
    measured=chord_loss(center.detach(),varied.reshape(1152,23),picked,base,pair,base,pair,torch.ones(23))
    expected=np.mean([np.mean((np.arange(1,24,dtype=np.float64)[None,:]*(cell+1)*np.arange(1,3)[:,None])**2) for cell in range(9)])
    assert float(measured)==float(expected)
    strata=strata_for(np.tile(np.arange(250,1269),3),np.repeat(np.arange(3),1019))
    assert [[len(ids) for ids in group] for group in strata]==[[100,819,100]]*3
    rng=torch.get_rng_state();rows,axes=sample_pairs(strata)
    next_rng=torch.get_rng_state();torch.set_rng_state(rng);repeat_rows,repeat_axes=sample_pairs(strata)
    assert torch.equal(rows,repeat_rows) and torch.equal(axes,repeat_axes) and torch.equal(torch.get_rng_state(),next_rng)
    for cell,ids in enumerate([ids for group in strata for ids in group]):
        assert np.isin(rows[cell*64:(cell+1)*64].numpy(),ids).all()
    assert int(axes.min())>=0 and int(axes.max())<23
    source=Path(__file__).parent;out=source.parent/'chord_objective_pure_tests_v2.json';assert not out.exists()
    result=dict(passed=True,center_gradient_connected=True,paired_and_nine_cell_mean_weight_exact=True,
        centered_matching_teacher_zero_loss=True,sampler_restored_rng_reproducible=True,all_strata_and_axes_valid=True,
        distinct_cell_axis_sign_reduction_exact=True,
        actual_model_evaluations=0,optimizer_updates=0,BFM_calls=0,physics_steps=0,
        source_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in source.glob('*.py')})
    out.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print('PURE TENSOR OBJECTIVE TESTS PASS: connected center gradients, equal cell weights, RNG sampling')

if __name__=='__main__':main()
