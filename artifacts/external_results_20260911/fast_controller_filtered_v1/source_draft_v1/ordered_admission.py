"""Fixed first-feasible candidate admission; forecasting supplied by the caller."""
import time
import numpy as np

CANDIDATE_NAMES=('primary','original_BFM','previous_applied','current_q')

def admit(candidates,forecast):
    if len(candidates)!=4:raise ValueError('Admission requires the four fixed ordered candidates.')
    prepared=[np.asarray(target,np.float64).copy() for target in candidates]
    if any(target.shape!=(23,) or not np.isfinite(target).all() for target in prepared):raise ValueError('Invalid admission candidate.')
    records=[];unique=[]
    for index,target in enumerate(prepared):
        duplicate=next((prior for prior in unique if np.array_equal(target,prepared[prior])),None)
        if duplicate is None:
            tick=time.perf_counter();witness=forecast(target.copy());elapsed=(time.perf_counter()-tick)*1000
            if type(witness.get('feasible')) is not bool:raise ValueError('Forecast must return an explicit Boolean feasible verdict.')
            record=dict(candidate=index,name=CANDIDATE_NAMES[index],target=target.copy(),duplicate_of=None,
                forecast_ms=elapsed,witness=witness,feasible=witness['feasible'])
            unique.append(index)
        else:
            prior=records[duplicate]
            record=dict(candidate=index,name=CANDIDATE_NAMES[index],target=target.copy(),duplicate_of=duplicate,
                forecast_ms=0.,witness=None,feasible=prior['feasible'])
        records.append(record)
        if record['feasible']:return index,target.copy(),records
    return None,None,records
