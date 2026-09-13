"""Draft fixed sampling and differentiable center reuse. Imported only by reviewed fit."""
import numpy as np
import torch

def strata_for(control,dataset):
    return [[np.flatnonzero((dataset==code)&mask) for mask in
        ((control>=250)&(control<350),(control>=350)&(control<1169),(control>=1169)&(control<1269))]
        for code in range(3)]

def sample_pairs(strata):
    rows=[];axes=[]
    for group in strata:
        for ids in group:
            # Exactly nine calls in dataset, then phase order; restored global RNG.
            product=torch.randint(0,len(ids)*23,(64,))
            rows.append(torch.from_numpy(ids)[product//23]);axes.append(product%23)
    return torch.cat(rows),torch.cat(axes)

def chord_loss(normalized_centers,normalized_probes,picked,base_centers,base_probes,
               target_centers,target_probes,span):
    """No detached center or additional center network evaluations."""
    center_delta=normalized_centers*span
    probe_delta=(normalized_probes*span).reshape(576,2,23)
    center_proposal=base_centers[picked]+center_delta[picked]
    probe_proposal=base_probes+probe_delta
    exact_teacher_change=target_probes-target_centers[picked,None]
    difference=((probe_proposal-center_proposal[:,None])-exact_teacher_change)/span
    square=difference**2
    return torch.stack([square[cell*64:(cell+1)*64].mean() for cell in range(9)]).mean()
