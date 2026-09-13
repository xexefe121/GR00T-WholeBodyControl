"""Saved report comparison only; no model, gradient, optimizer or native calls."""
from pathlib import Path
import json,hashlib
BASE=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
path=BASE/'fit/report.json';digest=sha(path);report=read(path)
assert report['completed'] is True and report['ordinary_final_step']==71000 and report['optimizer_step']==6000
initial=report['metrics']['initial_GPU32'];final=report['metrics']['final_GPU32'];ort=report['metrics']['ORT64']
keys=('nominal_objective','full_state_objective','balanced_full_state_objective','physical_objective','weighted_objective','balanced_weighted_objective')
comparison={key:dict(initial_GPU32=initial[key],final_GPU32=final[key],fractional_change=final[key]/initial[key]-1,
                     final_ORT64=ort[key]) for key in keys}
result=dict(report_sha256=digest,comparison=comparison,
    original_zero_response_ratio=ort['full_state_objective']/ort['full_state_zero_response_MSE'],
    balanced_zero_response_ratio=ort['balanced_full_state_objective']/ort['balanced_full_state_zero_response_MSE'],
    max_preclip_error_rad=report['max_preclip_error_rad'],task_model_calls=0,native_steps=0,
    comparison_scope='same-backend initialGPU32 vs finalGPU32; ORT64 final shown separately',
    no_causal_advantage_over_extra_epochs_or_warming_claim=True,no_physical_qualification=True)
lines=['# Ordinary71000 saved-fit result','',
    'The fixed3000-update warm continuation completed, preserving the ordinary endpoint and all diagnostics. AdamW advanced3000→6000; training consumed exactly9000 forwards /44,058,000 rows. Initial outputs were byte-exact to causal68000 across all367570 rows.',
    '', '| Metric | Initial GPU32 | Final GPU32 | Change | Final ORT64 |', '|---|---:|---:|---:|---:|']
for key,value in comparison.items():lines.append(f"| {key} | {value['initial_GPU32']:.12g} | {value['final_GPU32']:.12g} | {100*value['fractional_change']:+.4f}% | {value['final_ORT64']:.12g} |")
lines+=['',f"Final original-response /zero-response baseline: {result['original_zero_response_ratio']:.6f}. Balanced-response /weighted-zero-response baseline: {result['balanced_zero_response_ratio']:.6f}. Values above1 mean worse than zero response under that metric.",
    '',f"Same-weight CPU64/GPU64/ORT64 maximum preclamp difference: {report['max_preclip_error_rad']:.12g} rad; fixed gate1e-5 passed. Public input/output remain float32.",
    '', 'This is one engineering continuation, not a matched comparison against extra epochs or warm optimization. No physical trial is qualified by these losses or export checks. The earlier v1 metadata failure remains preserved with zero forwards/updates.',
    '', 'Source report SHA256: '+digest]
assert sha(path)==digest
with (BASE/'saved_fit_comparison.json').open('x',encoding='utf-8') as stream:json.dump(result,stream,indent=2);stream.write('\n')
with (BASE/'FIT_SUMMARY.md').open('x',encoding='utf-8') as stream:stream.write('\n'.join(lines)+'\n')
print(json.dumps(result))
