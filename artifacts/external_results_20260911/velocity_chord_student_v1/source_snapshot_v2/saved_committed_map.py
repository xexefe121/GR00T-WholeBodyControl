"""Exact saved plan arithmetic, extracted from the qualified core without importing a solver."""
import ast
from types import SimpleNamespace
import numpy as np
from chord_common import NEW, OLD, archive, read

CORE = NEW/'bfm_entry250_actual_oracle_v1/source_snapshot_v1/gear_sonic/utils/g1_true23_mjbatch_ilqr_core.py'
PRODUCER = OLD/'mjbatch_full_v1/walk003_v4_native323_allmargin_full_v1'

def difference_function():
    tree = ast.parse(CORE.read_text())
    pure = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in ('quat_mul','quat_log')]
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'Planner')
    pure.append(next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == 'difference'))
    scope = {'np':np}
    exec(compile(ast.fix_missing_locations(ast.Module(body=pure,type_ignores=[])),str(CORE),'exec'),scope)
    fake = SimpleNamespace(nq=30,nv=29,lin_q=np.r_[0:3,7:30],lin_v=np.r_[0:3,6:29],free=[(0,0)])
    return lambda nominal, actual: scope['difference'](fake,nominal[None],actual[None])[0]

def plans_for_dataset(name, trace):
    out = {}
    if name == 'old':
        saved = archive(PRODUCER/'trace.npz')
        for key in ('qpos','qvel'):
            np.testing.assert_array_equal(saved[key][:1270], trace[key][:1270])
        np.testing.assert_array_equal(saved['target'][:1269],trace['target'][:1269])
        for plan in read(PRODUCER/'plans.json'):
            start = plan['control']
            for c in range(start,start+plan['controls_committed']):
                if 250 <= c < 1269:
                    out[c] = dict(plan_control=start, local=c-start, accepted_update=-1,
                        planned_state=saved['planned_state'][c].copy(),
                        planned_target=saved['planned_target'][c].copy(), gain=saved['feedback_gain'][c].copy())
    else:
        expert = NEW/('student_actual_oracle_control1_resume1001_v1' if name=='query1' else 'bfm_entry250_actual_oracle_v1')
        for plan in read(expert/'nominal/plans.json'):
            start = plan['control']; count = plan['executed_controls']
            if start+count <= 250 or start >= 1269:
                continue
            saved = archive(expert/'nominal/plans'/('plan_%05d.npz'%start))
            accepted = int(any(it['accepted'] for it in plan['solver_feasibility']['iterations']))
            for c in range(start,start+count):
                if 250 <= c < 1269:
                    at = c-start
                    out[c] = dict(plan_control=start, local=at, accepted_update=accepted,
                        planned_state=saved['nominal_states'][at].copy(),
                        planned_target=saved['targets'][at].copy(), gain=saved['gains'][at].copy())
    assert set(out) == set(range(250,1269))
    return out

def committed_target(difference, plan_state, plan_target, gain, qpos, qvel, limits):
    # Preserve the full original matrix product and both clips; no column shortcut.
    actual = np.r_[qpos,qvel]
    tangent = difference(plan_state,actual)
    raw = gain @ tangent
    correction = np.clip(raw,-.1,.1)
    preclip = plan_target + correction
    target = np.clip(preclip,limits[:,0],limits[:,1])
    return target,raw,correction,preclip
