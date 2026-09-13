"""Independent saved-array/source binding only; no policy, simulator, or solver imports."""
from pathlib import Path
import hashlib
import json
import ast
import numpy as np

HERE = Path(__file__).resolve().parent
PILOT = HERE.parent / 'fast_controller_nominal_pilot_v1'
OLD = Path('E:/codex-artifacts/sonic23_teleop_six_hour_20260910')
TEACHER = OLD / 'bfm_online_intent_v2/walk003_terminal_bfm_yaw4_hybrid_v1'

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def load(path):
    with np.load(path, allow_pickle=False) as z:
        return {k:z[k].copy() for k in z.files}

def main():
    receipts = {}
    for version, name in [('v1','frozen_inputs.json'), ('v2','frozen_inputs_v2.json')]:
        receipt = json.loads((PILOT/name).read_text())
        source = PILOT / ('source_snapshot' if version=='v1' else 'source_snapshot_v2')
        sources = {k: sha(source/k) for k in receipt['source_sha256']}
        inputs = {k: sha(k) for k in receipt['input_sha256']}
        assert sources == receipt['source_sha256']
        assert inputs == receipt['input_sha256']
        receipts[version] = dict(receipt_sha256=sha(PILOT/name), source_sha256=sources,
                                 input_sha256=inputs, all_bindings_exact=True)
    labels = load(PILOT/'labels/labels.npz')
    teacher = load(TEACHER/'trace.npz')
    report = json.loads((PILOT/'labels/report.json').read_text())
    assert sha(PILOT/'labels/labels.npz') == report['labels_sha256']
    assert report['source_frozen_receipt_sha256'] == receipts['v1']['receipt_sha256']
    contract_path = next(Path(k) for k in receipts['v2']['input_sha256'] if k.endswith('/mjbatch_native23_inputs_v1/contract.json'))
    contract = json.loads(contract_path.read_text())
    n = 1269
    equality = {}
    for key, source_key in [('expert_target','target'), ('previous_action','previous_action'),
                            ('history','history'), ('state','state')]:
        np.testing.assert_array_equal(labels[key], teacher[source_key][:n])
        equality[key] = True
    for key, source_key in [('teacher_qpos','qpos'), ('teacher_qvel','qvel')]:
        np.testing.assert_array_equal(labels[key], teacher[source_key][:n+1])
        equality[key] = True
    np.testing.assert_array_equal(labels['control'], np.arange(n))
    np.testing.assert_array_equal(labels['source_frame'], np.arange(n)+11)
    np.testing.assert_array_equal(labels['residual_rad'], labels['expert_target']-labels['base_target'])
    normalized = ((labels['expert_target']-np.asarray(contract['default_q']))*
                  np.asarray(contract['kp'])/(.25*np.asarray(contract['training_effort']))).astype(np.float32)
    np.testing.assert_array_equal(normalized, teacher['action'][:n])
    np.testing.assert_array_equal(labels['previous_action'][0], np.zeros(23,np.float32))
    np.testing.assert_array_equal(labels['previous_action'][1:], normalized[:-1])
    assert all(np.isfinite(value).all() for value in labels.values())
    source_v1=PILOT/'source_snapshot'; source_v2=PILOT/'source_snapshot_v2'
    semantic_unchanged={}
    for name in ('quiet_metrics.py','terminal_yaw4_goal.py',
                 'gear_sonic/utils/g1_true23_bfm_seed_observations.py',
                 'gear_sonic/utils/g1_true23_mpc_student.py'):
        semantic_unchanged[name]=ast.dump(ast.parse((source_v1/name).read_text()))==ast.dump(ast.parse((source_v2/name).read_text()))
    assert all(semantic_unchanged.values())
    out=dict(kind='independent_readonly_student_source_and_label_audit',
             imports='stdlib and NumPy only; no simulator/policy/solver executed',
             receipts=receipts,label_trace_sha256=sha(PILOT/'labels/labels.npz'),
             teacher_trace_sha256=sha(TEACHER/'trace.npz'),
             label_generation_receipt_matches_preserved_v1=True,
             exact_teacher_rows=equality,exact_controls_and_source_frames=True,
             exact_residual_definition=True,exact_actual_target_action_normalization=True,
             exact_previous_action_recurrence=True,all_saved_label_arrays_finite=True,
             unchanged_goal_history_quiet_helpers=semantic_unchanged,
             source_controls=819,entry_acquisition_controls=350,return_controls=100,
             samples=1269,no_terminal_BFM_fit_labels=True,
             behavioral_quality_claim=False,hardware_authorized=False)
    (HERE/'source_and_label_audit.json').write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps({k:v for k,v in out.items() if k not in ('receipts',)}))

if __name__=='__main__':
    main()
