"""Compact saved-status inspection only; no controller/model/physics imports."""
import json
from pathlib import Path
BASE=Path(__file__).resolve().parent
result={}
for name in ('canonical_process_status.json','canonical_prefix250_parity.json','actual_query250_input_parity.json','query250_frozen_head_proposal_parity.json'):
    path=BASE/name
    if path.exists():result[name]=json.loads(path.read_text(encoding='utf-8-sig'))
for segment in ('nominal','post_lifecycle_hold_5s'):
    path=BASE/segment/'report.json'
    if path.exists():
        raw=json.loads(path.read_text())
        keys=('requested_controls','completed_full_controls','full_segment_completed','physics_steps','failure','not_run_reason',
            'range_excess_max','velocity_ratio_max','effort_ratio_max','engine_warning_counts','independent_accumulated_clock_max_error',
            'selected_forecast_actual_substeps_checked','private_forecasts','private_physics_steps','selected_candidate_counts',
            'ranked_attempted_policy_ms_p50_p95_max','ranked_attempted_policy_20ms_deadline_misses',
            'source_metrics','quiet_standing_diagnostic','trace_sha256','private_forecasts_sha256','candidate_costs_sha256','admission_ledger_sha256')
        result[segment]={key:raw[key] for key in keys if key in raw}
for stream in ('stdout','stderr'):
    path=BASE/('canonical_'+stream+'.log')
    if path.exists():
        raw=path.read_bytes();decoded=raw.decode('utf-16' if raw[:2] in (b'\xff\xfe',b'\xfe\xff') else 'utf-8-sig',errors='replace')
        lines=[line for line in decoded.splitlines() if line.strip()]
        result[stream+'_tail']=[line[:700] for line in lines[-2:]]
print(json.dumps(result,indent=2,allow_nan=False))
