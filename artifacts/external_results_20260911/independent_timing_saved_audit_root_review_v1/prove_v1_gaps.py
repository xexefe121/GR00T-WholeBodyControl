"""Independent synthetic regressions only; never imports producer/runtime."""
import hashlib,json,sys
from pathlib import Path
OUT=Path(__file__).resolve().parent
SOURCE=OUT.parent/'independent_plant_timing_saved_audit_v1/source_draft_v1'
sys.path.insert(0,str(SOURCE))
from test_timing_saved_math import fixture,check
from test_timing_sidecar_io import reserved
from timing_sidecar_io import reserved_capture
results={}
f=fixture();p={r['phase']:r for r in f['rows']}
p[2]['end_wall_before']=p[3]['start_wall_before']+5
p[2]['end_wall_after']=p[3]['start_wall_before']+6
results['sequential_sibling_wall_overlap_accepted']=check(f)['instrumentation_complete']
f=fixture();p={r['phase']:r for r in f['rows']}
p[11]['end_thread_cpu']=p[10]['end_thread_cpu']+1
results['child_thread_cpu_after_parent_end_accepted']=check(f)['instrumentation_complete']
f=fixture();p={r['phase']:r for r in f['rows']}
p[11]['end_process_cpu']=p[10]['end_process_cpu']+1
results['child_process_cpu_after_parent_end_accepted']=check(f)['instrumentation_complete']
a=reserved();a[2]['session']['stepper']['verification_attempts']=a[2]['session']['stepper']['returned']
results['attempt_misreported_as_return']=reserved_capture(*a)['actual_native_verification_returned']
assert all(results.values())
record=dict(proved_on_frozen_source_v1=True,results=results,task_arrays_opened=0,native_calls=0,model_calls=0,
 source_sha256={n:hashlib.sha256((SOURCE/n).read_bytes()).hexdigest() for n in ('timing_saved_math.py','timing_sidecar_io.py')},
 writer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
with (OUT/'v1_gap_proof.json').open('x',encoding='utf-8') as f:json.dump(record,f,indent=2)
print(json.dumps(results))
