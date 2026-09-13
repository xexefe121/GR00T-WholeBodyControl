"""Prepare immutable teacher-energy constants and preserve original loss sources."""
from pathlib import Path
import hashlib, json

BASE=Path(__file__).resolve().parent
NEW=BASE.parent
SOURCE=BASE/'source_draft_v1'
SOURCE.mkdir(exist_ok=True)
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
report_path=NEW/'direct_target_context_response_conditioning_v1/report.json'
assert sha(report_path)=='84a1fb497cdc0c6edf3b8db7be69a604dc4b1b7e73041b437f8c899d882f429c'
report=json.loads(report_path.read_text())
names=('root_position','root_rotation','joint_position','root_linear_velocity','root_angular_velocity','joint_velocity')
pins={report_path.as_posix():sha(report_path)}
energies=[];weights=[]
for condition in ('blinded','causal'):
    metrics_path=NEW/f'direct_target_causal_context_study_v2/fit/{condition}/ORT64_metrics.json'
    assert sha(metrics_path)==report['input_sha256'][metrics_path.as_posix()]
    pins[metrics_path.as_posix()]=sha(metrics_path)
    metrics=json.loads(metrics_path.read_text())
    rows=report['results'][condition]['group_comparison']
    assert tuple(r['group'] for r in rows)==names
    assert [r['zero_response_MSE'] for r in rows]==[r['zero_response_MSE'] for r in metrics['full_state_group_comparison']]
    if energies:assert energies==[r['zero_response_MSE'] for r in rows]
    else:
        energies=[r['zero_response_MSE'] for r in rows]
        weights=[r['illustrative_fixed_energy_equalization'] for r in rows]
    assert metrics['full_state_zero_response_MSE']==0.00019286493749569113
assert all(v>0 for v in energies+weights)
original=NEW/'direct_target_causal_context_study_v2/source_snapshot_v1'
expected={'full_state_objective.py':'0fa1d2fca4f47b1ce1049976ecf8d6771b6b64949c018f3b813d191173b9fed2',
          'direct_objective.py':'396e4032d4ffb5ae29ed3c135e6a552b6b9958498e8a128733120e3c0fc19d97',
          'full_state_contract.py':'331d4ad4b4481df9c6c80ea7de73eccd0fff95c99aa53b71c24e4750d2336ad3'}
for name,digest in expected.items():
    path=original/name;assert sha(path)==digest
    with (SOURCE/name).open('xb') as stream:stream.write(path.read_bytes())
    assert sha(SOURCE/name)==digest;pins[path.as_posix()]=digest
constants='''"""Fixed teacher-only group energies; original54 cells use dataset/phase/group order."""
GROUP_NAMES = %r
ZERO_RESPONSE_ENERGIES = %r
MEAN_ZERO_RESPONSE_ENERGY = 0.00019286493749569113
GROUP_WEIGHTS = %r
CELL_GROUPS = tuple(range(6)) * 9
CONDITIONING_REPORT_SHA256 = %r
WEIGHT_RULE = 'mean_six_teacher_group_energies_over_group_energy'
''' % (names,tuple(energies),tuple(weights),sha(report_path))
with (SOURCE/'balance_contract.py').open('x') as stream:stream.write(constants)
with (BASE/'energy_source.json').open('x') as stream:
    json.dump(dict(group_names=names,zero_response_energies=energies,mean_zero_response_energy=.00019286493749569113,
        group_weights=weights,rule='Emean/Eg',weight_source='audited teacher zero-response energy; no fitted prediction determines any weight',
        input_sha256=pins,original_source_sha256=expected,task_prediction_arrays_read=0,task_checkpoint_reads=0,
        task_model_calls=0,task_gradient_calls=0,optimizer_updates=0,native_steps=0),stream,indent=2);stream.write('\n')
print(json.dumps(dict(energy_source_sha256=sha(BASE/'energy_source.json'),original_modules_exact=3)))
