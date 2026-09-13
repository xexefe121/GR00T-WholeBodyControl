"""Prepare unchanged bounded launch mechanics for a future width512 release."""
import difflib,hashlib,json
from pathlib import Path
BASE=Path(__file__).resolve().parent
OLD=BASE.parent/'direct_target_causal_response_evaluation_v1'
NAMES=('runtime_inventory.py','freeze_final_package.py','prepare_review_configuration.py','prepare_bound_launcher.py',
       'diagnostic_verdict.py','verify_completed_stage.py','test_launch_helpers.py','test_release_helpers.py')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def one(s,a,b):
    assert s.count(a)==1,(a,s.count(a))
    return s.replace(a,b)
for name in NAMES:
    with (BASE/name).open('xb') as f:f.write((OLD/name).read_bytes())
path=BASE/'runtime_inventory.py';s=path.read_text()
s=one(s,'reviewed71000 response-balanced context runtime source','reviewed81000 width512 context runtime source')
s=one(s,'explicit_response71000_fixed_runtime_inventory','explicit_width512_81000_fixed_runtime_inventory')
path.write_text(s,encoding='utf-8',newline='\n')
path=BASE/'freeze_final_package.py';s=path.read_text()
for a,b in [('warm balanced71000','width512 warm81000'),("FIT=NEW/'direct_target_causal_response_balanced_student_v2'","FIT=NEW/'direct_target_causal_width512_student_v1'"),
            ('ordinary_final_step=71000','ordinary_final_step=81000'),
            ('ordinary71000_warm_balanced_context_same_weight_fp64_export','ordinary81000_width512_warm_balanced_context_same_weight_fp64_export'),
            ('direct_absolute_target_1323_causal_response_balanced','direct_absolute_target_1323_causal_width512'),
            ('architecture=[1323,256,256,23]','architecture=[1323,512,512,23]'),
            ('one_ordinary71000_response_balanced_canonical_evaluation','one_ordinary81000_width512_canonical_evaluation')]:s=one(s,a,b)
path.write_text(s,encoding='utf-8',newline='\n')
path=BASE/'prepare_review_configuration.py';s=one(path.read_text(),'warm balanced71000','width512 warm81000');path.write_text(s,encoding='utf-8',newline='\n')
path=BASE/'test_release_helpers.py';s=path.read_text()
s=one(s,"b['architecture']==[1323,256,256,23] and b['ordinary_final_step']==71000","b['architecture']==[1323,512,512,23] and b['ordinary_final_step']==81000")
path.write_text(s,encoding='utf-8',newline='\n')
unchanged=[n for n in NAMES if sha(BASE/n)==sha(OLD/n)]
assert set(unchanged)=={'prepare_bound_launcher.py','diagnostic_verdict.py','verify_completed_stage.py','test_launch_helpers.py'}
diff=''.join(''.join(difflib.unified_diff((OLD/n).read_text().splitlines(True),(BASE/n).read_text().splitlines(True),fromfile='preserved71000/'+n,tofile='width81000/'+n)) for n in NAMES if n not in unchanged)
with (BASE/'launch_helper_changes.diff').open('x',encoding='utf-8') as f:f.write(diff)
result=dict(preparation_only=True,helper_sha256={n:sha(BASE/n) for n in NAMES},original_helper_sha256={n:sha(OLD/n) for n in NAMES},
 byte_identical_helpers=unchanged,changes_sha256=sha(BASE/'launch_helper_changes.diff'),writer_sha256=sha(Path(__file__)),
 actual_fit_or_export_read=False,model_calls=0,native_steps=0,actual_binding_created=False)
with (BASE/'launch_helper_derivation.json').open('x',encoding='utf-8') as f:json.dump(result,f,indent=2);f.write('\n')
print(json.dumps({'helpers':8,'byte_identical_helpers':4,'derivation_sha256':sha(BASE/'launch_helper_derivation.json')}))
