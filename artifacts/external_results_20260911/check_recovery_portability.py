import copy,json,sys,math
from pathlib import Path
base=Path(__file__).resolve().parent
sys.path.insert(0,str(base/'direct_target_width251_evaluation_v2/source_draft_v1'))
from recovery_release import recovery_request_identity,selected_rates
r=json.loads((base/'direct_target_width251_student_v1/training_request.json').read_text())
actual=r['learning_rate_values'];computed=selected_rates()
recovery_request_identity(r)
bad=copy.deepcopy(r);bad['learning_rate_values'][9999]=math.nextafter(actual[9999],math.inf)
try:recovery_request_identity(bad)
except AssertionError:pass
else:raise AssertionError('Changed schedule accepted')
print(json.dumps(dict(platform=sys.platform,actual_schedule_accepted=True,one_ulp_change_rejected=True,
    old_recomputed_mismatches=sum(a!=b for a,b in zip(actual,computed)),
    largest_recomputed_difference=max(abs(a-b) for a,b in zip(actual,computed)),model_calls=0,native_steps=0)))
