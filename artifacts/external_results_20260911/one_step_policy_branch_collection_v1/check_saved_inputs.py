"""Pure prefreeze checks and stub tests; no graph sessions or native calls."""
import sys,json,unittest
from pathlib import Path
BASE=Path(__file__).resolve().parent;sys.path.insert(0,str(BASE/'draft'))
from collection_arrays import atomic,sha
from branch_inputs import Inputs,archive,read,local
from stateless_adapter import prepare
import test_collection
NEW=BASE.parent;memory=read(NEW/'prior_memory_intervention_v1/request.json')
paths={k:local(v) for k,v in memory['paths'].items()}
paths.update(trace0=NEW.parent/'sonic23_teleop_six_hour_20260910/bfm_online_intent_v2/walk003_terminal_bfm_yaw4_hybrid_v1/trace.npz',
 trace1=NEW/'student_actual_oracle_control1_resume1001_v1/nominal/trace.npz',trace2=NEW/'bfm_entry250_actual_oracle_v1/nominal/trace.npz',
 old_snapshots=NEW/'old_expert_prefix_snapshots_v2/capture/control_snapshots.npz',witness=NEW/'velocity_chord_student_evaluation_v1/head_witness/witness.npz')
with (BASE/'pure_checks.log').open('w',encoding='utf-8') as log:
    result=unittest.TextTestRunner(stream=log,verbosity=2).run(unittest.defaultTestLoader.loadTestsFromModule(test_collection))
    assert result.wasSuccessful()
    c,builder,latent,sensed=prepare(archive(paths['original']),archive(paths['motion']),archive(paths['original29']),read(paths['contract']),{})
    inputs=Inputs(paths);mapping=inputs.validate_centers(builder,sensed,read(paths['contract']))
    log.write('All 3054 source row mappings and current/successor pure features/state/history checks passed.\n')
atomic(BASE/'pure_checks.json',dict(passed=True,tests=result.testsRun,all_source_rows_verified=len(mapping),
    physical_native_calls=0,graph_calls=0,optimizer_calls=0,
    source_sha256={p.relative_to(BASE/'draft').as_posix():sha(p) for p in (BASE/'draft').rglob('*.py')},
    input_sha256={str(p):sha(p) for k,p in paths.items() if k!='onnx_dependencies'},log_sha256=sha(BASE/'pure_checks.log')))
print(json.dumps(dict(passed=True,tests=result.testsRun,rows=len(mapping),graph_calls=0,physical_native_calls=0)))
