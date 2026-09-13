"""Read durable saved status and reports only; no controller or physics imports."""
import json
from pathlib import Path
BASE=Path(__file__).resolve().parent
result={}
for name in ('canonical_process_status.json','canonical_prefix250_parity.json','actual_query250_input_parity.json',
    'query250_frozen_head_proposal_parity.json','nominal/report.json','post_lifecycle_hold_5s/report.json'):
    path=BASE/name
    if path.exists():result[name]=json.loads(path.read_text(encoding='utf-8-sig'))
for stream in ('stdout','stderr'):
    path=BASE/('canonical_'+stream+'.log')
    if path.exists():
        raw=path.read_bytes();decoded=raw.decode('utf-16' if raw[:2] in (b'\xff\xfe',b'\xfe\xff') else 'utf-8-sig',errors='replace')
        lines=[line for line in decoded.splitlines() if line.strip()]
        result[stream+'_tail']=lines[-2:]
print(json.dumps(result,indent=2,allow_nan=False))
