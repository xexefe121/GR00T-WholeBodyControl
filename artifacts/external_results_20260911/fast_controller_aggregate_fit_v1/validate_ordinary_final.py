"""Read-only ordinary-final identity checks; no inference, fit or simulation."""
from pathlib import Path
import hashlib
import json
import numpy as np
import onnx
from onnx import numpy_helper
import torch

BASE = Path(__file__).resolve().parent
PRIOR = BASE.parent / 'fast_controller_continued_fit_v1'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    destination = BASE / 'final_export_validation.json'
    assert not destination.exists()
    fit = BASE / 'fit'
    report = json.loads((fit / 'report.json').read_text())
    assert report['network_training_complete'] and report['steps'] == 60000 and report['additional_updates'] == 40000
    assert report['physical_evaluations'] == 0 and report['expert_queries'] == 0
    assert report['ONNX_vs_Torch_max_delta_rad'] < 1e-5
    for name, expected in report['checkpoints'].items():
        assert sha(fit / name) == expected
    checkpoint = torch.load(fit / 'student_head.pt', map_location='cpu', weights_only=True)
    previous = torch.load(PRIOR / 'fit/student_head.pt', map_location='cpu', weights_only=True)
    assert checkpoint['completed_steps'] == 60000 and checkpoint['additional_updates'] == 40000
    assert len(checkpoint['optimizer_state']['state']) == 6
    assert all(int(state['step']) == 60000 for state in checkpoint['optimizer_state']['state'].values())
    assert all(torch.isfinite(value).all() for state in checkpoint['optimizer_state']['state'].values()
               for value in state.values() if torch.is_tensor(value))
    for name in ('feature_mean', 'feature_std', 'joint_span'):
        assert torch.equal(checkpoint[name], previous[name])
    assert all(torch.isfinite(value).all() for value in checkpoint['actor_state'].values())
    model = onnx.load(str(fit / 'student_head.onnx'))
    onnx.checker.check_model(model)
    initializers = {item.name: numpy_helper.to_array(item) for item in model.graph.initializer}
    for name, key in [('mean', 'feature_mean'), ('std', 'feature_std'), ('span', 'joint_span')]:
        np.testing.assert_array_equal(initializers[name], checkpoint[key].numpy())
    for number, layer in enumerate((0, 2, 4)):
        np.testing.assert_array_equal(initializers['w%d' % number], checkpoint['actor_state']['%d.weight' % layer].numpy().T)
        np.testing.assert_array_equal(initializers['b%d' % number], checkpoint['actor_state']['%d.bias' % layer].numpy())
    metrics = json.loads((fit / 'full_fit_metrics.json').read_text())
    assert [m['global_step'] for m in metrics] == list(range(20000, 60001, 1000))
    assert all(m['physical_evaluations'] == 0 and m['checkpoint_selection'] is False for m in metrics)
    logs = json.loads((fit / 'training_log.json').read_text())
    assert [m['global_step'] for m in logs] == list(range(20100, 60001, 100))
    assert len(metrics[-1]['fixed_pairs']) == len(metrics[0]['fixed_pairs'])
    result = dict(kind='ordinary_final60000_export_read_only_validation', pass_=True,
        ordinary_final_global_step=60000, additional_updates=40000, optimizer_parameter_states_at60000=6,
        all_model_tensors_finite=True, original_norm_span_exact=True, ONNX_initializer_checkpoint_identity_exact=True,
        recorded_ONNX_vs_Torch_max_delta_rad=report['ONNX_vs_Torch_max_delta_rad'],
        diagnostic_steps_exact=True, fixed_pair_count=len(metrics[-1]['fixed_pairs']),
        old_fit_rmse_rad=metrics[-1]['datasets']['old']['all']['applied_target_rmse_rad'],
        new_fit_rmse_rad=metrics[-1]['datasets']['new']['all']['applied_target_rmse_rad'],
        query_control1_fit_rmse_rad=metrics[-1]['actual_student_query_control1']['target_rmse_rad'],
        files_sha256={str(path): sha(path) for path in [fit/'report.json', fit/'student_head.pt',
            fit/'student_head.onnx', fit/'teacher_fit.npz', fit/'restoration20000_parity.json',
            fit/'optimization_completed.json', fit/'full_fit_metrics.json', BASE/'frozen_inputs_v2.json', Path(__file__)]},
        validation_inference_calls=0, validation_optimizer_calls=0, validation_physics_steps=0,
        canonical_rollout_started=False, hardware_authorized=False)
    destination.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps({k: result[k] for k in ('pass_', 'ordinary_final_global_step', 'old_fit_rmse_rad', 'new_fit_rmse_rad', 'query_control1_fit_rmse_rad')}), flush=True)


if __name__ == '__main__':
    main()
