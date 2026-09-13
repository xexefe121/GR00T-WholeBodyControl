"""Independent pure-array witnesses for previously found validation holes."""
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np

repo=Path('Z:/codex/GR00T-WholeBodyControl-sonic-transfer-23dof')
path=repo/'gear_sonic/utils/g1_true23_mjbatch_model.py'
source_bytes=path.read_bytes()
scope={'__file__':str(path)}
exec(compile(source_bytes,str(path),'exec'),scope)
model=SimpleNamespace(nq=30,nv=29,nu=23,opt=SimpleNamespace(timestep=.002),
                      jnt_range=np.tile([-2.,2.],(24,1)))
predicate=scope['Native23Feasibility'](model,dict(native_velocity=np.ones(23),native_effort=np.ones(23)))
q=np.zeros((2,30));q[:,2]=.8;q[:,3]=1;dq=np.zeros((2,29))
checks={}
for name,kwargs in [('force_Nx1',dict(force=np.zeros((2,1)))),
                    ('force_singleton_wrong_batch',dict(force=np.zeros((1,23)))),
                    ('warning_Nx1',dict(warning=np.zeros((2,1)))),
                    ('time_Nx1',dict(time=np.zeros((2,1)),expected_time=0))]:
    try:
        predicate.assess(q,dq,**kwargs)
    except ValueError:
        checks[name+'_rejected']=True
    else:
        raise AssertionError(name+' incorrectly accepted')
assert not predicate.assess(q,dq,force=np.zeros((2,23)),warning=np.zeros((2,8)),
                            time=np.zeros(2),expected_time=0)[0].any()
checks['well_formed_quiet_state_accepted']=True
q[:,3]=0
assert predicate.assess(q,dq)[0].all()
checks['zero_quaternion_rejected']=True
q[:,3]=1+1e-9
assert predicate.assess(q,dq)[0].all()
checks['quaternion_norm_matches_oracle_1e10_tolerance']=True
q[:,3]=1
assert predicate.assess(q,dq,time=np.full(2,np.nan),expected_time=0)[0].all()
assert predicate.assess(q,dq,force=np.full((2,23),np.nan))[0].all()
checks.update(nonfinite_clock_rejected=True,nonfinite_force_rejected=True)
report=dict(scope='Pure array validation only; no MjModel, MjData, mj_step or solver call',
            source_file=str(path),source_sha256=hashlib.sha256(source_bytes).hexdigest(),
            checks=checks,contract=predicate.contract())
assert source_bytes==path.read_bytes(), 'source changed during pure-array witness'
Path(__file__).with_name('predicate_model_snapshot.py').write_bytes(source_bytes)
output=Path(__file__).with_name('predicate_checks.json')
output.write_text(json.dumps(report,indent=2))
print(json.dumps(report))
