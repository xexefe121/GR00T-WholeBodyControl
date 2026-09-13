"""Exercise cost archive shapes and censored rows using saved arrays only."""
import ast
import hashlib
import json
from pathlib import Path
import tempfile
import numpy as np
BASE=Path(__file__).resolve().parent.parent
SOURCE=BASE/'source_draft_v1/evaluate_filtered_student.py'

def main():
    tree=ast.parse(SOURCE.read_text());node=next(node for node in ast.walk(tree) if isinstance(node,ast.FunctionDef) and node.name=='save_costs')
    scope={'np':np};exec(compile(ast.Module(body=[node],type_ignores=[]),str(SOURCE),'exec'),scope);save=scope['save_costs']
    with np.load(BASE.parent/'filtered_all_candidates132_v1/results/cost_knots.npz',allow_pickle=False) as archive:
        keys=('state','features','residual','component_cost','state_cost','input_cost','target_reference')
        full={key:archive[key][0].copy() for key in keys}
        full['state_goal_frame']=archive['state_goal_frames'][0].copy();full['input_goal_frame']=archive['input_goal_frames'][0].copy()
        case=72;count=int(archive['knot_available'][case].sum())
        partial={key:archive[key][case,:count].copy() if key not in ('input_cost','target_reference') else archive[key][case].copy() for key in keys}
        partial['state_goal_frame']=archive['state_goal_frames'][case,:count].copy();partial['input_goal_frame']=archive['input_goal_frames'][case].copy()
    with tempfile.TemporaryDirectory(prefix='cost_persistence_') as tmp:
        dest=Path(tmp);save(dest,[],[])
        with np.load(dest/'candidate_costs.npz',allow_pickle=False) as value:
            assert value['residual'].shape==(0,6,157) and value['input_cost'].shape==(0,5)
            assert value['state_goal_frame'].dtype==np.int64 and value['cost_available'].dtype==np.bool_
        metadata=[dict(control=250,candidate=0),dict(control=268,candidate=0),dict(control=268,candidate=1)]
        save(dest,[full,partial,None],metadata)
        with np.load(dest/'candidate_costs.npz',allow_pickle=False) as value:
            np.testing.assert_array_equal(value['cost_available'],[True,True,False])
            assert value['state_knot_available'][0].sum()==6 and value['state_knot_available'][1].sum()==count and not value['state_knot_available'][2].any()
            for key,array in full.items():np.testing.assert_array_equal(value[key][0,:len(array)],array)
            for key,array in partial.items():np.testing.assert_array_equal(value[key][1,:len(array)],array)
            assert np.isnan(value['state_cost'][1,count:]).all() and np.isnan(value['residual'][2]).all()
            assert np.all(value['state_goal_frame'][2]==-1)
    report=dict(pass_=True,driver_sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        checks=['empty cost shapes and dtypes','full case exact','censored case available knots exact',
            'unscored fault row preserves explicit unavailable mask','padding never represented as a real cost'],
        physics_steps=0,actor_calls=0,model_imports=0)
    (BASE/'cost_persistence_test_report.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8');print(json.dumps(report))

if __name__=='__main__':main()
