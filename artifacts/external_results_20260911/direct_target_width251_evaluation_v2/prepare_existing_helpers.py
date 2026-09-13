"""Reuse existing witness and full-motion launch path for the recovery endpoint."""
import ast,hashlib,json
from pathlib import Path
BASE=Path(__file__).resolve().parent
OLD=BASE.parent/'direct_target_causal_width512_evaluation_v1'
NAMES=('runtime_inventory.py','freeze_final_package.py','prepare_review_configuration.py','prepare_bound_launcher.py',
       'diagnostic_verdict.py','verify_completed_stage.py','test_launch_helpers.py','test_release_helpers.py')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def one(s,a,b):
    assert s.count(a)==1,(a,s.count(a))
    return s.replace(a,b)
for name in NAMES:
    with (BASE/name).open('xb') as f:f.write((OLD/name).read_bytes())
path=BASE/'runtime_inventory.py';s=path.read_text().replace('reviewed81000 width512 context runtime source','reviewed91000 width512 recovery runtime source').replace('explicit_width512_81000_fixed_runtime_inventory','explicit_width512_recovery91000_fixed_runtime_inventory');path.write_text(s,encoding='utf-8',newline='\n')
path=BASE/'freeze_final_package.py';s=path.read_text()
for old,new in [('width512 warm81000','width512 recovery91000'),("FIT=NEW/'direct_target_causal_width512_student_v1'","FIT=NEW/'direct_target_width251_student_v1'"),
    ('ordinary_final_step=81000','ordinary_final_step=91000'),('ordinary81000_width512_warm_balanced_context_same_weight_fp64_export','ordinary91000_width512_recovery_context_same_weight_fp64_export'),
    ('direct_absolute_target_1323_causal_width512','direct_absolute_target_1323_causal_width512_recovery'),('one_ordinary81000_width512_canonical_evaluation','one_ordinary91000_width512_recovery_canonical_evaluation')]:s=one(s,old,new)
s=one(s,"energy_source=Path(request['subjects']['energy_source']['path']),", "energy_source=Path(request['subjects']['energy_source']['path']),\n        **{role:Path(request['subjects'][role]['path']) for role in ('recovery_rows','collection_report','collection_request','collection_qualification','collection_source_review','consistency_report','warm_restore_review')},")
path.write_text(s,encoding='utf-8',newline='\n')
path=BASE/'prepare_review_configuration.py';s=one(path.read_text(),'width512 warm81000','width512 recovery91000');path.write_text(s,encoding='utf-8',newline='\n')
path=BASE/'test_release_helpers.py';s=one(path.read_text(),"b['architecture']==[1323,512,512,23] and b['ordinary_final_step']==81000","b['architecture']==[1323,512,512,23] and b['ordinary_final_step']==91000");path.write_text(s,encoding='utf-8',newline='\n')
for name in NAMES:ast.parse((BASE/name).read_text(encoding='utf-8-sig'))
unchanged=[n for n in NAMES if sha(BASE/n)==sha(OLD/n)]
assert set(unchanged)=={'prepare_bound_launcher.py','diagnostic_verdict.py','verify_completed_stage.py','test_launch_helpers.py'}
with (BASE/'launch_helper_derivation.json').open('x',encoding='utf-8') as f:
    json.dump(dict(helper_sha256={n:sha(BASE/n) for n in NAMES},original_helper_sha256={n:sha(OLD/n) for n in NAMES},
        byte_identical_helpers=unchanged,model_calls=0,native_steps=0,actual_binding_created=False),f,indent=2)
print('Existing eight evaluation helpers adapted; four execution helpers unchanged.')
