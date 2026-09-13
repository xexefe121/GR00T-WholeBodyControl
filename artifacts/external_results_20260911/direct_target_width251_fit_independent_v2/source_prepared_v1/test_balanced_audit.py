"""Synthetic saved arrays only; no trained model, forwards or optimizers."""
import copy
from pathlib import Path
import numpy as np
import pytest
from audit_balanced_math import GROUPS,energy_weights,balanced_metrics,warm_initial_errors,ledger_expectations
from audit_release import release_paths
from audit_context_math import WEIGHT

def energy_fixture():
    e=np.array([1,2,4,8,16,32],np.float64)*.001
    return dict(group_names=list(GROUPS),rule='Emean/Eg',zero_response_energies=e.tolist(),
        mean_zero_response_energy=float(e.mean()),group_weights=(e.mean()/e).tolist())

def metric_fixture(energy):
    cells=[]
    for d in range(3):
        for p in range(3):
            for g,e in enumerate(energy['zero_response_energies']):
                cells.append(dict(dataset=d,phase=p,tangent_group=g,response_MSE=e,
                    first24_response_MSE=e*2,odd_response_MSE=e*.75,even_response_MSE=e*.25,zero_response_MSE=e))
    f=float(np.mean([c['response_MSE'] for c in cells]))
    return dict(nominal_objective=.3,full_state_objective=f,physical_objective=.7,
        weighted_objective=(.3+WEIGHT*f)+.7,full_state_cells=cells)

def test_teacher_energy_and_unweighted_metrics_preserved():
    e=energy_fixture();w=energy_weights(e);m=metric_fixture(e);old=copy.deepcopy(m)
    result=balanced_metrics(m,w,e)
    assert m==old
    for k in old:assert result[k]==old[k]
    assert np.isclose(result['balanced_full_state_objective'],e['mean_zero_response_energy'],rtol=1e-15)
    assert np.isclose(result['balanced_full_state_zero_response_MSE'],e['mean_zero_response_energy'],rtol=1e-15)
    assert [c['weight'] for c in result['balanced_full_state_cells']]==list(w)*9

@pytest.mark.parametrize('damage',['energy','weight','group_order','cell_order','wrong_teacher'])
def test_bad_balance_subject_or_order_rejected(damage):
    e=energy_fixture();m=metric_fixture(e)
    if damage=='energy':e['zero_response_energies'][0]=0
    if damage=='weight':e['group_weights'][0]*=2
    if damage=='group_order':e['group_names'].reverse()
    if damage=='cell_order':m['full_state_cells'].reverse()
    if damage=='wrong_teacher':m['full_state_cells'][0]['zero_response_MSE']+=1
    with pytest.raises(AssertionError):balanced_metrics(m,energy_weights(e),e)

def test_original_and_balanced_loss_ledgers_distinguish_layout():
    n=np.tile(np.linspace(.01,.17,15,dtype=np.float32),(2,1)).astype(np.float64)
    f=np.tile(np.arange(1,55,dtype=np.float64)/100,(2,1));p=np.full((2,9),.13,np.float64)
    w=energy_weights(energy_fixture());r=ledger_expectations(n,f,p,w)
    np.testing.assert_array_equal(r['nominal'],n.astype(np.float32).mean(axis=1).astype(np.float64))
    np.testing.assert_array_equal(r['weighted_cells'],f*np.tile(w,9))
    assert not np.array_equal(r['weighted_cells'],f*np.repeat(w,9))
    np.testing.assert_array_equal(r['original_total'],(r['nominal']+WEIGHT*f.mean(axis=1))+p.mean(axis=1))
    np.testing.assert_array_equal(r['balanced_total'],(r['nominal']+WEIGHT*r['weighted_cells'].mean(axis=1))+p.mean(axis=1))
    assert np.all(r['original_total']!=r['balanced_total'])

def state_fixture():
    names=['0.weight','0.bias','2.weight','2.bias','4.weight','4.bias']
    actor={name:np.full((2,1323) if i==0 else (2,),i+1,np.float32) for i,name in enumerate(names)}
    opt=dict(state={i:dict(step=np.array(3000,np.float32),exp_avg=a*.01,exp_avg_sq=a*.03) for i,a in enumerate(actor.values())},param_groups=[dict(params=list(range(6)),lr=1e-6)])
    source=dict(actor_state=actor,optimizer_state=opt,feature_mean=np.zeros(1323,np.float32),feature_std=np.ones(1323,np.float32),
        rng={'synthetic':np.array([3,7,11],np.uint8)},condition='causal',ordinary_final_step=68000,optimizer_step=3000,context_blinded=False)
    initial=copy.deepcopy({k:source[k] for k in ('actor_state','optimizer_state','feature_mean','feature_std')});initial['rng_after_restoration']=copy.deepcopy(source['rng'])
    return initial,source

def test_exact_warm_source_and_initial_state():
    init,source=state_fixture();assert warm_initial_errors(init,source)==[]

@pytest.mark.parametrize('damage',['context','moments','step','RNG','normalization','both_fractional_step'])
def test_warm_corruption_rejected(damage):
    init,source=state_fixture()
    if damage=='context':init['actor_state']['0.weight'][:,1000:]=0
    if damage=='moments':init['optimizer_state']['state'][0]['exp_avg'].fill(0)
    if damage=='step':init['optimizer_state']['state'][0]['step'][...]=0
    if damage=='RNG':init['rng_after_restoration']['synthetic'][0]=17
    if damage=='normalization':init['feature_mean'][1000]=1
    if damage=='both_fractional_step':
        for value in (init,source):value['optimizer_state']['state'][0]['step'][...]=3000.5
    assert warm_initial_errors(init,source)

def test_exact_single_release_roles():
    extra=('recovery_rows','collection_report','collection_request','collection_qualification','collection_source_review','consistency_report','warm_restore_review','selected_protocol')
    request=dict(subjects={key:dict(path='/synthetic/'+key) for key in ('coefficient_source','checkpoint','full_state_root_audit','full_state_root_owner','energy_source')+extra},full_state_paths=dict(request='/synthetic/generation_request',report='/synthetic/generation_report'))
    roles=release_paths(Path('/synthetic/study'),request)
    assert set(roles)==set('fit_report checkpoint head normalization training_manifest training_request export_manifest coefficient source_checkpoint full_state_generation_request full_state_generation_report full_state_data_audit full_state_data_owner shared_manifest context_alignment energy_source recovery_evidence'.split())|set(extra)
    assert len(roles)==25 and roles['checkpoint'].name=='student_head.pt' and roles['fit_report'].parent.name=='fit'
    assert not any('paired' in str(p) or 'blinded' in str(p) for p in roles.values())
