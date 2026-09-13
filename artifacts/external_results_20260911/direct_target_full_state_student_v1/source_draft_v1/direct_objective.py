"""No base/residual terms. Promote predictions before finite subtraction."""
import torch

def nominal_loss(prediction,labels,cells):
    square=(prediction-labels)**2
    per_cell=torch.stack([square[ids].mean() for ids in cells])
    return per_cell.mean(),per_cell

def velocity_loss(nominal_prediction,probe_prediction,nominal_indices,teacher_change_normalized):
    center=nominal_prediction[nominal_indices].to(torch.float64)
    endpoint=probe_prediction.to(torch.float64).reshape(576,2,23)
    square=((endpoint-center[:,None])-teacher_change_normalized)**2
    per_cell=torch.stack([square[k*64:(k+1)*64].mean() for k in range(9)])
    return per_cell.mean(),per_cell

def physical_loss(nominal_prediction,endpoint_prediction,nominal_successors,teacher_change_normalized,cells,counts):
    center=nominal_prediction[nominal_successors].to(torch.float64)
    endpoint=endpoint_prediction.to(torch.float64)
    square=((endpoint-center)-teacher_change_normalized)**2
    if len(cells)!=9 or len(counts)!=9:raise ValueError('Expected nine physical cells.')
    per_cell=torch.stack([square[ids].sum()/(n*23) for ids,n in zip(cells,counts)])
    return per_cell.mean(),per_cell
