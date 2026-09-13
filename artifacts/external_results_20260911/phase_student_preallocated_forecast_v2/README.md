# Reusable private native forecast: fixed-case qualification

All 232 fixed cases passed. Every qpos, qvel, command torque, actual actuator force, time, warning count and warning last-info sample is byte-exact against the independently implemented deepcopy oracle. First failures, aggregate physical metrics and the original root forecast reports also match exactly. Both complete trace archives have identical SHA256 dae782bfec6153186d9397e3dfb9de97e86bb4a68c07d0e716d67cb95b2ef630. There were 6,480 private native steps per method, zero policy/optimizer calls and zero actual plant steps. Every caller's complete 291-element integration state and warnings stayed unchanged.

## Fixed API

```python
from native_forecast import NativeForecast
forecast = NativeForecast(native_model, contract, maximum_controls=5)
report, trace_views = forecast.predict(actual_data, bounded_target, horizon_controls=5)
```

Construct once for the pinned native model. `bounded_target` must already be a finite float64 NumPy array of shape (23,), inside the native target limits. The helper never repairs a target. `horizon_controls` must be a Python int from 1 through the fixed maximum. It predicts a constant target for 10 native 2 ms steps per requested control, stopping at the first unchanged strict predicate failure.

`report['feasible']` is the whole-segment verdict; `first_failure` contains the exact witness. The report owns its metadata. Returned trace arrays are views into reused storage: copy any evidence that must survive the next candidate forecast. The object is single-call, single-thread state; do not share concurrent calls.

One private MjData plus full-state, PD and trace workspaces are allocated at construction. Each call restores all integration fields and warnings, forwards privately to rebuild derived data, then restores full integration again. No per-call MjData/deepcopy or full trace workspace is allocated. Python metadata, scalar objects, small predicate arrays and returned views still allocate; this is not a claim of zero heap allocation.

## Cost on this one pass

The fixed order alternated component/oracle and oracle/component across cases. There was one pass, no best-of timing or parameter selection. Timing includes full state and warning copying, private initialization, all strict checks, trace writing, result metadata, owned output copying and component caller-immutability checks.

| Constant-target horizon | Reusable p50 / p95 / max, ms | Independent deepcopy oracle p50 / p95 / max, ms |
|---|---:|---:|
| 20 ms | 2.106 / 4.104 / 9.469 | 44.549 / 65.169 / 105.081 |
| 100 ms | 9.123 / 17.646 / 21.475 | 52.590 / 76.629 / 80.911 |

Concurrent work was uncontrolled. Four serial candidates plus actor inference can exceed a 20 ms control budget. These measurements qualify neither realtime execution nor hardware. Passing a 100 ms constant-target forecast does not establish recursive feasibility or source tracking. No fallback was selected and no connected controller was run.

## Preserved setup failure

Version 1 stopped before its first forecast because the MuJoCo Python binding requires a writable mj_setState array, whereas the immutable cases were read-only. Version 2 changes the benchmark runner only: it copies each fixed integration into one preallocated writable buffer. Component bytes, all 232 candidates and all horizons are unchanged. Version 1 source, frozen receipt and failure outputs remain intact.

The completed report is `results/report.json` (SHA256 67e516f3cfd059438677580b8a7b575b183af270327f3143a13748b62201e478). The pre-execution request is `request.json` (SHA256 e2ff7f22e9f96598383af7a9ef2ac444fffa61514f003aee676958ce783c3fd6). Component SHA256 is 81af01274f55c703850cb3ec3801bb41912e3f3786440af5099b769b7bd28824.
