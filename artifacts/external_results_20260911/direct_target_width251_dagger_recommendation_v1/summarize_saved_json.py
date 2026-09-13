"""Saved JSON/proximity only. No task arrays, model imports, or calls."""
import hashlib
import json
import statistics
from pathlib import Path

BASE=Path(__file__).resolve().parent
NEW=BASE.parent
CONS=NEW/'direct_target_width251_consistency_v1/actual_v1/results_v1'
paths=[CONS/'report.json',CONS/'proximity.jsonl',NEW/'direct_target_width251_first_target_comparison_v1/report.json',
       NEW/'direct_target_width512_convergence_review_v1/report.json',NEW/'direct_target_width512_convergence_review_v1/diagnosis.md']
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert sha(paths[0])=='2ddf917a98d77268e55fe38151a94a3208428d4fe334a613820583c450a6cbaf'
report=json.loads(paths[0].read_text())
assert sha(paths[1])==report['outputs']['proximity.jsonl']
rows=[json.loads(line) for line in paths[1].read_text().splitlines()]
assert len(rows)==72*4
def describe(values):return dict(minimum=min(values),median=statistics.median(values),maximum=max(values))
groups=[]
for view in sorted({r['view'] for r in rows}):
    for phase in range(3):
        selected=[r for r in rows if r['view']==view and r['query_phase']==phase]
        assert len(selected)==24
        groups.append(dict(view=view,phase=phase,queries=24,
            full_normalized_RMS=describe([r['normalized_full_RMS'] for r in selected]),
            current_normalized_RMS=describe([r['normalized_current_RMS'] for r in selected]),
            target_RMSE_rad=describe([r['target_RMSE_rad'] for r in selected]),
            target_max_abs_rad=describe([r['target_max_abs_rad'] for r in selected]),
            candidate_feedback_clipped=sum(r['candidate_feedback_clipped'] is True for r in selected),
            query_feedback_clipped=sum(r['query_feedback_clipped'] is True for r in selected),
            zero_history_distance=sum(r['normalized_history_RMS']==0. for r in selected)))
first=next(r for r in rows if r['query_control']==251 and r['view']=='full_global')
result=dict(saved_json_only=True,proximity_groups=groups,first_query_nearest=first,
    first_target_comparison=json.loads(paths[2].read_text()),
    recommendation=dict(selected=False,updates=10000,ordinary_start=81000,ordinary_final=91000,
        optimizer_start=16000,optimizer_final=26000,recovery_coefficient=0.2,
        original_nominal_coefficient=1.0,original_physical_coefficient=1.0,original_response_coefficient=1.8188207859141674,
        per_old_nominal_cell_coefficient='1/15',per_new_phase_coefficient='0.2/3 = 1/15',
        schedule='existing10000x864 byte-exact prefix; no regeneration or wrap',
        rates='indices0..249 inclusive linear1e-6 to1e-5; indices250..9999 inclusive cosine1e-5 to1e-6',
        new_gradient_calibration=False,new_optimizer=False,training_calls=40000,training_rows=157040000,
        diagnostic_Torch_calls=5764,diagnostic_Torch_rows=1474352,diagnostic_ORT_calls=1441,diagnostic_ORT_rows=368588),
    input_sha256={p.as_posix():sha(p) for p in paths},task_array_loads=0,checkpoint_loads=0,
    model_calls=0,native_calls=0,gradient_calls=0,optimizer_updates=0)
with (BASE/'report.json').open('x') as f:json.dump(result,f,indent=2,allow_nan=False)
print(json.dumps(dict(groups=groups,report_sha256=sha(BASE/'report.json'))))
