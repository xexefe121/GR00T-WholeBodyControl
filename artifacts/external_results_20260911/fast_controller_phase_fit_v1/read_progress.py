"""Read saved receipts/logs only; no model imports or inference."""
import hashlib
import json
from pathlib import Path
BASE=Path(__file__).resolve().parent
def read(path):return json.loads(path.read_text(encoding='utf-8-sig'))
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
result={}
for phase in ('fit','canonical'):
    status=BASE/(phase+'_process_status.json')
    if status.exists():result[phase]=read(status)
    log=BASE/(phase+'_stdout.log')
    if log.exists():
        raw=log.read_bytes();text=raw.decode('utf-16' if raw[:2] in (b'\xff\xfe',b'\xfe\xff') else 'utf-8-sig',errors='replace')
        lines=[line for line in text.splitlines() if line.strip().startswith('{')]
        if lines:
            try:result[phase+'_last_log']=json.loads(lines[-1])
            except json.JSONDecodeError:result[phase+'_last_log']='partial JSON log write'
for name in ('restoration60000_parity.json','optimization_completed.json','report.json','fit_failure.json'):
    path=BASE/'fit'/name
    if path.exists():
        report=read(path)
        if name=='report.json':report={key:report[key] for key in ('steps','additional_updates','full_objective_improved',
            'initial_nine_cell_objective','final_nine_cell_objective','ONNX_vs_Torch_max_delta_rad','export_parity_passed',
            'rollout_numerical_prerequisites_pass','elapsed_seconds','checkpoints')}
        result[name]=report
for name in ('nominal','post_lifecycle_hold_5s'):
    path=BASE/name/'report.json'
    if path.exists():result[name]=dict(report_sha256=sha(path),report=read(path))
print(json.dumps(result,indent=2,allow_nan=False))
