"""Read-only final65000 checkpoint/export validation, without inference or updates."""
from pathlib import Path
import hashlib
import json
import math
import numpy as np
import onnx
from onnx import numpy_helper
import torch

BASE=Path(__file__).resolve().parent
PRIOR=BASE.parent/'fast_controller_aggregate_fit_v1'

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def same(a,b):
    if torch.is_tensor(a):return torch.equal(a,b)
    if isinstance(a,dict):return a.keys()==b.keys() and all(same(a[k],b[k]) for k in a)
    if isinstance(a,(list,tuple)):return len(a)==len(b) and all(same(x,y) for x,y in zip(a,b))
    return a==b

def main():
    destination=BASE/'final_export_validation.json'
    assert not destination.exists()
    fit=BASE/'fit';report=json.loads((fit/'report.json').read_text())
    assert report['network_training_complete'] and report['steps']==65000 and report['additional_updates']==5000
    assert report['physical_evaluations']==report['expert_queries']==0
    assert report['export_parity_passed'] and report['rollout_numerical_prerequisites_pass']
    assert report['full_objective_improved'] and 0<=report['final_nine_cell_objective']<report['initial_nine_cell_objective']
    assert math.isfinite(report['ONNX_vs_Torch_max_delta_rad']) and report['ONNX_vs_Torch_max_delta_rad']<1e-5
    for name,expected in report['checkpoints'].items():assert sha(fit/name)==expected
    checkpoint=torch.load(fit/'student_head.pt',map_location='cpu',weights_only=True)
    previous=torch.load(PRIOR/'fit/student_head.pt',map_location='cpu',weights_only=True)
    assert checkpoint['completed_steps']==65000 and checkpoint['additional_updates']==5000
    assert len(checkpoint['optimizer_state']['state'])==6
    assert all(int(state['step'])==65000 for state in checkpoint['optimizer_state']['state'].values())
    assert all(torch.isfinite(v).all() for s in checkpoint['optimizer_state']['state'].values() for v in s.values() if torch.is_tensor(v))
    assert all(group['lr']==3e-6 and group['weight_decay']==1e-5 for group in checkpoint['optimizer_state']['param_groups'])
    for name in ('feature_mean','feature_std','joint_span'):assert torch.equal(checkpoint[name],previous[name])
    assert same(checkpoint['rng'],previous['rng'])
    assert all(torch.isfinite(value).all() for value in checkpoint['actor_state'].values())
    model=onnx.load(str(fit/'student_head.onnx'));onnx.checker.check_model(model)
    initializers={item.name:numpy_helper.to_array(item) for item in model.graph.initializer}
    for name,key in [('mean','feature_mean'),('std','feature_std'),('span','joint_span')]:
        np.testing.assert_array_equal(initializers[name],checkpoint[key].numpy())
    for number,layer in enumerate((0,2,4)):
        np.testing.assert_array_equal(initializers['w%d'%number],checkpoint['actor_state']['%d.weight'%layer].numpy().T)
        np.testing.assert_array_equal(initializers['b%d'%number],checkpoint['actor_state']['%d.bias'%layer].numpy())
    metrics=json.loads((fit/'full_fit_metrics.json').read_text());logs=json.loads((fit/'training_log.json').read_text())
    assert [m['global_step'] for m in metrics]==list(range(60000,65001,250))
    assert all(m['physical_evaluations']==0 and m['checkpoint_selection'] is False for m in metrics)
    assert [m['global_step'] for m in logs]==list(range(60001,65001))
    expected_rates=np.asarray([3e-6+.5*(3e-5-3e-6)*(1+math.cos(math.pi*k/4999)) for k in range(5000)])
    with np.load(fit/'training_arrays.npz',allow_pickle=False) as arrays:
        np.testing.assert_array_equal(arrays['global_step'],np.arange(60001,65001))
        np.testing.assert_array_equal(arrays['learning_rate'],expected_rates)
        np.testing.assert_array_equal(arrays['learning_rate'],[m['learning_rate'] for m in logs])
        np.testing.assert_array_equal(arrays['training_objective'],[m['training_objective'] for m in logs])
        assert np.isfinite(arrays['training_objective']).all()
    assert len(metrics[-1]['fixed_pairs'])==len(metrics[0]['fixed_pairs'])==243
    assert metrics[0]['nine_cell_normalized_residual_objective']==report['initial_nine_cell_objective']
    assert metrics[-1]['nine_cell_normalized_residual_objective']==report['final_nine_cell_objective']
    with np.load(fit/'teacher_fit.npz',allow_pickle=False) as arrays:
        np.testing.assert_array_equal(arrays['control'],np.tile(np.arange(250,1269),3))
        np.testing.assert_array_equal(arrays['source_frame'],np.tile(np.arange(261,1280),3))
        np.testing.assert_array_equal(arrays['dataset'],np.repeat(np.arange(3),1019))
        assert arrays['predicted_delta'].shape==(3057,23) and np.isfinite(arrays['predicted_delta']).all()
    result=dict(kind='ordinary_final65000_export_read_only_validation',pass_=True,
        ordinary_final_global_step=65000,additional_updates=5000,optimizer_parameter_states_at65000=6,
        all_model_tensors_finite=True,original_norm_span_exact=True,RNG_unchanged=True,
        ONNX_initializer_checkpoint_identity_exact=True,all5000_learning_rates_exact=True,
        recorded_ONNX_vs_Torch_max_delta_rad=report['ONNX_vs_Torch_max_delta_rad'],
        initial_nine_cell_objective=report['initial_nine_cell_objective'],final_nine_cell_objective=report['final_nine_cell_objective'],
        diagnostic_steps_exact=True,fixed_pair_count=243,
        fit_rmse_rad={name:metrics[-1]['datasets'][name]['all']['applied_target_rmse_rad'] for name in ('old','query1','query250')},
        query250_fit_rmse_rad=metrics[-1]['actual_query250']['target_rmse_rad'],
        files_sha256={str(path):sha(path) for path in [fit/name for name in ('report.json','student_head.pt',
            'student_head.onnx','teacher_fit.npz','training_arrays.npz','restoration60000_parity.json',
            'optimization_completed.json','full_fit_metrics.json','training_log.json')]+[BASE/'frozen_inputs_v2.json',Path(__file__)]},
        validation_inference_calls=0,validation_optimizer_calls=0,validation_physics_steps=0,
        canonical_rollout_started=False,hardware_authorized=False)
    destination.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps({k:result[k] for k in ('pass_','ordinary_final_global_step','fit_rmse_rad','query250_fit_rmse_rad')}),flush=True)

if __name__=='__main__':main()
