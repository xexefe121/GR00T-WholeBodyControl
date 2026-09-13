"""Derive profile conclusions from saved events/arrays only; no dynamics."""
import hashlib
import json
from pathlib import Path
import numpy as np

p=Path(__file__).resolve().parent
r=json.loads((p/'report.json').read_text())
e=json.loads((p/'events.json').read_text())
t=r['stage_times']; total=t['planning_total']['inclusive_ms']
stage={name:dict(milliseconds=t[name]['inclusive_ms'],percent=100*t[name]['inclusive_ms']/total,calls=t[name]['calls'])
       for name in ['seed_scoring.recorded','seed_scoring.shifted','seed_scoring.fresh','fresh_seed_generation','linearize','expand','backward','line_search_rollout']}
clone_checks={}
with np.load(p/'solve_arrays.npz',allow_pickle=False) as f:
    for key in f.files:
        if key.startswith('seed_scoring_'):
            a=f[key]
            if a.ndim==3:
                same=all(np.array_equal(a[:,j],a[:,0]) for j in range(9))
            elif a.ndim==1:
                same=bool(np.array_equal(a,np.repeat(a[0],9)))
            else:
                raise AssertionError((key,a.shape))
            clone_checks[key]=dict(shape=list(a.shape),all_nine_lanes_bit_exact=same)
            assert same
    expected_shapes={'planned_states':(31,59),'planned_targets':(30,23),'gains':(30,23,58)}
    for key,shape in expected_shapes.items():
        assert f[key].shape==shape and np.isfinite(f[key]).all()
    output_shapes={key:list(f[key].shape) for key in expected_shapes}

def ticks(text):
    # Linux aggregate user,nice,system,idle,iowait,irq,softirq,steal.
    return np.array(list(map(int,text.splitlines()[0].split()[1:9])),np.int64)
d=ticks(r['cpu_after']['proc_stat'])-ticks(r['cpu_before']['proc_stat'])
busy=float((d.sum()-d[3]-d[4])/d.sum())
seed_ms=sum(t[name]['inclusive_ms'] for name in ['seed_scoring.recorded','seed_scoring.shifted','seed_scoring.fresh'])
first_steps=[x['inclusive_ms'] for x in e if x['name']=='linear_batch.step'][::2]
other_steps=[x['inclusive_ms'] for x in e if x['name']=='linear_batch.step'][1::2]
parent_ids=[x['parent'] for x in e]
assert all(x['exclusive_ms']>=-1e-9 for x in e)
top=next(x for x in e if x['name']=='planning_total')
assert abs(sum(x['exclusive_ms'] for x in e)-top['inclusive_ms'])<1e-8
result=dict(stage_breakdown=stage,total_ms=total,
    counted_instrumented_calls=len(e),exclusive_accounting_matches_total=True,
    seed_duplicate_arrays=clone_checks,full_solve_output_shapes=output_shapes,
    dynamics_derivative_work=dict(state_tangent_dimensions=58,target_dimensions=23,knots=30,lanes=2460,
         derivative_evaluations=5,native_steps_per_lane=10,total_native_steps=123000,
         measured_first_step_total_ms=sum(first_steps),measured_remaining_nine_steps_total_ms=sum(other_steps)),
    approximate_WSL_aggregate_busy_fraction_during_plan=busy,
    approximate_busy_logical_CPUs_during_plan=busy*r['runtime']['cpu_count'],
    seed_scoring_upper_bound_eliminating_entire_stage_ms=seed_ms,
    lower_bound_even_with_free_seed_scoring_ms=total-seed_ms,
    mathematical_equivalence_candidate='Unguided seed rollout uses no alpha or feedback term: one trajectory, cost and warning delta can replace nine identical lanes; retain lane0 arithmetic, clipping, forward and warmstart semantics.',
    candidate_implemented=False,candidate_speedup_measured=False,
    exactness_scope='This profile matches saved first-five planned states/targets/gains and final scalar cost. Original full uncommitted horizon is not archived for an independent full-horizon comparison.',
    timing_scope='Single isolated planning call, exact saved control3200; not a timing distribution or live deadline qualification.',
    report_sha256=hashlib.sha256((p/'report.json').read_bytes()).hexdigest(),
    events_sha256=hashlib.sha256((p/'events.json').read_bytes()).hexdigest(),
    solve_arrays_sha256=hashlib.sha256((p/'solve_arrays.npz').read_bytes()).hexdigest(),
    analysis_script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
(p/'analysis.json').write_text(json.dumps(result,indent=2,allow_nan=False))
print(json.dumps(result,indent=2))
