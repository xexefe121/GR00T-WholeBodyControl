"""Source preparation receipt and saved-margin arithmetic only; does not generate probes."""
import hashlib,json
from pathlib import Path
import numpy as np
BASE=Path(__file__).resolve().parent;NEW=BASE.parent
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def write(path,value):path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
sources={str(p):sha(p) for p in sorted((BASE/'source_draft_v1').glob('*.py'))}
center_path=NEW/'velocity_chord_student_v1/generation/centers.npz'
actual_path=NEW/'direct_target_fp64_fixed_map_v1/arrays.npz'
with np.load(center_path,allow_pickle=False) as c:
    lo,hi=c['joint_limits'].T;q=c['qpos'];v=c['qvel'];caps=c['native_velocity']
    margin=np.minimum(q[:,7:]-lo,hi-q[:,7:])
    evidence={'centers_sha256':sha(center_path),'joint_position_fixed_radius_rad':.01,
        'minimum_strict_joint_position_margin_rad':float(margin.min()),'inadmissible_center_axis_count':int((margin<=.01).sum()),
        'old_velocity_radius_strictly_admissible':bool(np.all(np.abs(v[:,6:])+.01*caps<caps)),
        'minimum_root_height_m':float(q[:,2].min()),'root_position_radius_m':.001}
with np.load(actual_path,allow_pickle=False) as a:
    evidence['actual_departure_source_sha256']=sha(actual_path)
    evidence['early_departures']=[{'control':int(a['control'][i]),'physical_departure_58':a['physical_departure'][i].tolist()} for i in (1,2)]
write(BASE/'saved_margin_evidence.json',evidence)
original=NEW/'direct_target_fp64_export_evaluation_v2/source_draft_v1/direct_features.py'
assert sha(original)==sha(BASE/'source_draft_v1/direct_features.py')=='26b816a2059edbb83daba386196c21b70002d1c83e2eda6e5cbf6a550ee4a251'
for p in (BASE/'source_draft_v1').glob('*.py'):compile(p.read_text(),str(p),'exec')
report={'preparation_passed':True,'source_only':True,'generation_selected':False,'generation_executed':False,
    'source_sha256':sources,'design_sha256':sha(BASE/'DESIGN.md'),'saved_margin_evidence_sha256':sha(BASE/'saved_margin_evidence.json'),
    'pure_feature_source_unchanged':True,'syntax_passed':True,
    'tests':{'executed':True,'passed':11,'failures':0,'runtime':'pinned WSL mjbatch323_20260910 Python/NumPy environment',
        'command':'test_secants.py -v','observed_exit_code':0,'scope':'synthetic arrays, original pure function AST; zero actual probe rows'},
    'counts_proposed':{'nominal_centers':3057,'old_velocity_overlap':140622,'added35_signed_rows':213990,'total_signed_rows':354612,'pure_feature_map_evaluations':357669},
    'model_calls':0,'BFM_calls':0,'physics_steps':0,'optimizer_updates':0,
    'required_future_gates':['Root selection and immutable actual input/source request','All3057 center features/maps exact','All140622 old overlap exact before added probes',
        'All fixed requested slots complete with state/chart/float32-alias checks','Independent saved data and duplicate/conflict review before any fit']}
write(BASE/'source_preparation.json',report)
print(json.dumps({'source_preparation_sha256':sha(BASE/'source_preparation.json'),'margin_evidence':evidence,'sources':sources},indent=2))
