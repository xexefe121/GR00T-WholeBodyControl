"""Separate full1018 three-phase target loss; original objectives are untouched."""
import numpy as np
import torch
from direct_objective import nominal_loss

PHASE_COUNTS=(99,819,100)

def phase_cells(phase,control):
    if phase.shape!=(1018,) or phase.dtype!=np.int8 or control.shape!=(1018,) or control.dtype!=np.int64:
        raise ValueError('Exact recovery metadata schema')
    if not np.array_equal(control,np.arange(251,1269,dtype=np.int64)) or not np.array_equal(phase,np.repeat(np.arange(3,dtype=np.int8),PHASE_COUNTS)):
        raise ValueError('Exactly1018 chronological recovery controls and phases')
    return [np.flatnonzero(phase==p) for p in range(3)]

def recovery_loss(prediction,labels,cells):
    if prediction.shape!=(1018,23) or labels.shape!=prediction.shape or prediction.dtype!=torch.float32 or labels.dtype!=torch.float32:
        raise ValueError('Full1018 original float32 supervised targets')
    if tuple(len(ids) for ids in cells)!=PHASE_COUNTS:raise ValueError('Three recovery phase cells')
    return nominal_loss(prediction,labels,cells)
