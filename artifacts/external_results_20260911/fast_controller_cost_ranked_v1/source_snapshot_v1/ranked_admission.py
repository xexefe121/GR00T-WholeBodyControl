"""Evaluate all four fixed candidates; rank only complete feasible exact costs."""
import time
import numpy as np

CANDIDATE_NAMES=('primary','original_BFM','previous_applied','current_q')

def admit(candidates,forecast):
    if len(candidates)!=4:raise ValueError('Exactly four fixed candidates are required.')
    prepared=[np.asarray(target,np.float64).copy() for target in candidates]
    if any(target.shape!=(23,) or not np.isfinite(target).all() for target in prepared):raise ValueError('Invalid candidate target.')
    records=[];best=None;best_cost=None
    for index,target in enumerate(prepared):
        duplicate=next((prior for prior in range(index) if np.array_equal(target,prepared[prior])),None)
        tick=time.perf_counter();witness=forecast(index,target.copy());elapsed=(time.perf_counter()-tick)*1000
        if type(witness.get('feasible')) is not bool:raise ValueError('Explicit native feasibility is required.')
        scoring=witness['tracking_cost'];eligible=witness['feasible']
        if eligible and (witness['physics_steps']!=50 or scoring['feasible_for_cost_ranking'] is not True):raise ValueError('Feasible candidate lacks its complete fixed-prefix score.')
        value=scoring['five_control_prefix_state_and_input_sum'] if eligible else None
        if eligible and (value is None or not np.isfinite(value)):raise ValueError('Nonfinite eligible score.')
        records.append(dict(candidate=index,name=CANDIDATE_NAMES[index],target=target.copy(),duplicate_of=duplicate,
            duplicate_still_evaluated=duplicate is not None,forecast_ms=elapsed,witness=witness,
            feasible=eligible,eligible_prefix_cost=value))
        # Strictly smaller only: exact ties preserve the original first-in-order candidate.
        if eligible and (best is None or value<best_cost):best=index;best_cost=value
    return (None,None,records) if best is None else (best,prepared[best].copy(),records)
