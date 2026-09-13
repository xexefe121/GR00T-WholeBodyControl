"""Pure IEEE-754 clock regression; no simulator or controller calls."""
import hashlib
import json
from pathlib import Path

DT=.002
TOL=1e-10
STEPS=65300


def witness(start=0., fault=None):
    expected=actual=float(start)
    first_ideal_failure=first_accumulated_failure=None
    maximum_accumulated_error=0.
    event_step=37000
    for step in range(1,STEPS+1):
        # Expected time is independent and never assigned from actual time.
        expected += DT
        if fault=='missing_step' and step==event_step:
            pass
        else:
            actual += DT
        if fault=='extra_step' and step==event_step:
            actual += DT
        if fault=='reset' and step==event_step:
            actual=0.
        if fault=='small_cumulative_drift':
            actual+=1e-12
        accumulated_error=abs(actual-expected)
        maximum_accumulated_error=max(maximum_accumulated_error,accumulated_error)
        ideal_error=actual-(start+step*DT)
        if first_ideal_failure is None and abs(ideal_error)>TOL:
            first_ideal_failure=dict(step=step,actual_time=actual,ideal_time=start+step*DT,error=ideal_error)
        if first_accumulated_failure is None and accumulated_error>TOL:
            first_accumulated_failure=dict(step=step,actual_time=actual,expected_time=expected,error=actual-expected)
    return dict(start_time=start,steps=STEPS,fault=fault,
                first_old_ideal_formula_failure=first_ideal_failure,
                first_independent_accumulation_failure=first_accumulated_failure,
                maximum_accumulated_error_seconds=maximum_accumulated_error,
                final_ideal_roundoff_diagnostic=actual-(start+STEPS*DT))


normal=witness()
assert normal['first_old_ideal_formula_failure']['step']==60370
assert normal['first_independent_accumulation_failure'] is None
assert normal['maximum_accumulated_error_seconds']==0.
shifted=witness(start=123.456)
assert shifted['first_independent_accumulation_failure'] is None
faults={fault:witness(fault=fault) for fault in ['reset','missing_step','extra_step','small_cumulative_drift']}
for fault in ['reset','missing_step','extra_step']:
    assert faults[fault]['first_independent_accumulation_failure']['step']==37000
assert faults['small_cumulative_drift']['first_independent_accumulation_failure'] is not None
report=dict(scope='Pure floating-point regression only; no physics or optimization',
            timestep_seconds=DT,unchanged_comparison_tolerance_seconds=TOL,
            expected_clock_rule='Initialize once; add dt independently per scheduled step; never reset from observed clock',
            normal=normal,nonzero_start=shifted,faults=faults,
            script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
Path(__file__).with_name('clock_accumulation_regression.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report))
