"""Freeze one evidence-based protocol; no data, checkpoint or model execution."""
import ast,hashlib,json,math
from pathlib import Path
from datetime import datetime,timezone
BASE=Path(__file__).resolve().parent;NEW=BASE.parent
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
rate_source=NEW/'direct_target_causal_width512_student_v1/source_prepared_v1/width512.py'
assert sha(rate_source)=='48ff62b384628a57c50dabd2bc55d8164ea2da41bc3e1b3538b08c7b06820ae7'
tree=ast.parse(rate_source.read_text());function=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='proposed_rate')
ns={'math':math};exec(compile(ast.Module(body=[function],type_ignores=[]),str(rate_source),'exec'),ns)
rates=[ns['proposed_rate'](i) for i in range(10000)]
assert rates[0]==rates[-1]==1e-6 and abs(rates[249]-1e-5)<1e-20 and abs(rates[250]-1e-5)<1e-20
subjects={
 'source_review':BASE/'root_review.json',
 'first_target_comparison':NEW/'direct_target_width251_first_target_comparison_v1/report.json',
 'collection_report':NEW/'direct_target_width251_collection_v1/actual_v1/results_v1/report.json',
 'consistency_report':NEW/'direct_target_width251_consistency_v1/actual_v1/results_v1/report.json',
 'recommendation':NEW/'direct_target_width251_dagger_recommendation_v1/report.json',
 'rate_source':rate_source}
record=dict(configuration_selected=True,actual_fit_execution_selected=False,selected_utc=datetime.now(timezone.utc).isoformat(),
 updates=10000,ordinary_start_step=81000,ordinary_final_step=91000,optimizer_start_step=16000,optimizer_final_step=26000,
 learning_rate_values=rates,learning_rate=dict(ramp_updates=250,ramp_inclusive=[1e-6,1e-5],cosine_updates=9750,cosine_inclusive=[1e-5,1e-6]),
 recovery_coefficient=0.2,response_schedule='unchanged10000_prefix_no_wrap',
 old_nominal_coefficient=1.0,old_physical_coefficient=1.0,old_balanced_response_coefficient=1.8188207859141674,
 training_forward_calls=40000,training_forward_rows=157040000,diagnostic_calls_per_backend=1441,diagnostic_rows_per_backend=368588,
 rationale='Each new phase has coefficient0.2/3=1/15, matching each original nominal phase; preserve old feedback constraints while adding qualified fresh recovery supervision.',
 limitations=['Dataset weighting, not measured gradient balancing or optimality.',
 'Nearby old/new labels differ; no proof that a single connected expert trajectory covers later student departures.',
 'Full numerical release and original lifecycle/hold simulation checks remain required.'],
 subjects={k:dict(path=p.as_posix(),sha256=sha(p)) for k,p in subjects.items()},
 no_extra_calibration_or_scouting=True,checkpoint_selection=False,automatic_retry=False,
 model_calls=0,native_steps=0,optimizer_updates_executed=0,writer_sha256=sha(__file__))
out=BASE/'selected_protocol.json'
with out.open('x',encoding='utf-8') as f:json.dump(record,f,indent=2);f.write('\n')
print(json.dumps(dict(configuration_selected=True,sha256=sha(out),updates=10000,recovery_coefficient=0.2,actual_fit_execution_selected=False)))
