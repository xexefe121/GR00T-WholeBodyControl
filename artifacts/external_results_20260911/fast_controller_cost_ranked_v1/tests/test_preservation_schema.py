"""Exercise actual serialization functions using saved arrays, without runtime imports."""
import ast
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import numpy as np

BASE=Path(__file__).resolve().parent.parent
SOURCE=BASE/'source_draft_v1/evaluate_filtered_student.py'

def extract(path,name,namespace):
    tree=ast.parse(path.read_text(encoding='utf-8'))
    selected=[node for node in ast.walk(tree) if isinstance(node,ast.FunctionDef) and node.name==name]
    assert len(selected)==1
    exec(compile(ast.Module(body=selected,type_ignores=[]),str(path),'exec'),namespace)
    return namespace[name]

def main():
    with np.load(BASE.parent/'fast_controller_phase_fit_v1/nominal/trace.npz',allow_pickle=False) as raw:
        old={key:raw[key].copy() for key in raw.files}
    helpers={'np':np}
    convert=extract(SOURCE,'trace_arrays',helpers)
    new=extract(BASE/'source_draft_v1/evaluate_phase_student.py','new_trace',helpers)
    initial=SimpleNamespace(qpos=old['qpos'][0],qvel=old['qvel'][0],time=old['physics_time'][0],
        warning=SimpleNamespace(number=old['physics_warning_counts'][0],lastinfo=old['physics_warning_lastinfo'][0]))
    empty=new(initial)
    for key in ('raw_bfm_action','proposed_target','proposed_action','applied_normalized_action','applied_delta',
        'primary_applied_unchanged','selected_candidate','candidate_count','forecast_count','admission_ms','control_loop_ms'):empty[key]=[]
    result=convert(empty)
    assert result['target'].shape==(0,23) and result['qpos'].shape==(1,30) and result['qvel'].shape==(1,29)
    assert result['control_integration_before'].shape==(0,291) and result['control_history_before'].shape==(0,300)
    assert result['physics_torque'].shape==(0,23) and result['physics_warning_counts'].shape==(1,8)
    assert result['state'].dtype==np.float32 and result['global_control'].dtype==np.int64
    assert result['physics_warning_counts'].dtype==np.int32 and result['primary_applied_unchanged'].dtype==np.bool_
    # The same serializer must leave the saved nonempty trace values and dtypes intact.
    fields=list(new(initial));saved={key:old[key] for key in fields}
    restored=convert(saved)
    for key in fields:
        np.testing.assert_array_equal(restored[key],old[key]);assert restored[key].dtype==old[key].dtype
    # A rejected attempt accidentally mixed into actual command rows is an explicit schema error.
    misaligned=dict(empty);misaligned['control_previous_action_before']=[old['previous_action'][250]]
    try:convert(misaligned)
    except AssertionError:pass
    else:raise AssertionError('Uncommitted extra control row accepted.')
    save=extract(SOURCE,'save_forecasts',helpers)
    with tempfile.TemporaryDirectory(prefix='filtered_saved_schema_') as tmp:
        dest=Path(tmp);save(dest,[],[])
        with np.load(dest/'private_forecasts.npz',allow_pickle=False) as value:
            for key,width in {'target':23,'physics_qpos':30,'physics_qvel':29,'physics_torque':23,
                'physics_actuator_force':23,'warning_counts':8,'warning_lastinfo':8}.items():assert value[key].shape==(0,width)
            assert value['state_offsets'].dtype==value['step_offsets'].dtype==np.int64
            assert value['warning_counts'].dtype==value['warning_lastinfo'].dtype==np.int32
        before={key[14:]:value for key,value in old.items() if key.startswith('final_history_')}
        scope=dict(np=np,dest=dest,proposal={key:old[key][250] for key in ('target','features','state','history','previous_action')},
            control=250,before=old['control_integration_before'][250],warning_before=old['physics_warning_counts'][2500],
            info_before=old['physics_warning_lastinfo'][2500],qpos_before=old['qpos'][250],qvel_before=old['qvel'][250],
            history_before=np.concatenate([before[key].reshape(-1) for key in sorted(before)]),
            previous_before=old['previous_action'][250],recorded_before=250,named_history_before=before)
        reject=extract(SOURCE,'save_rejected',scope);reject()
        with np.load(dest/'rejected_proposal.npz',allow_pickle=False) as value:
            assert value['actual_history_before'].shape==(300,) and value['actual_recorded_controls_before']==250
            assert value['integration_before'].shape==(291,) and value['warning_before'].shape==(8,)
            for key,wanted in before.items():np.testing.assert_array_equal(value['actual_history_before_'+key],wanted)
            np.testing.assert_array_equal(value['actual_previous_action_before'],old['previous_action'][250])
        # A fault before a policy proposal still preserves the complete measured snapshot.
        scope['proposal']=None;reject()
        with np.load(dest/'rejected_proposal.npz',allow_pickle=False) as value:assert 'actual_history_before_actions' in value.files
    report=dict(pass_=True,driver_sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        checks=['zero-command native shapes and dtypes','saved nonempty trace unchanged','extra uncommitted row rejected',
        'zero-forecast shapes and dtypes','rejection full named and flat history, prior, clock, integration and warnings',
        'pre-proposal fault full snapshot'],runtime_imports=0,actor_calls=0,physics_steps=0)
    (BASE/'preservation_schema_test_report.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(report))

if __name__=='__main__':main()
