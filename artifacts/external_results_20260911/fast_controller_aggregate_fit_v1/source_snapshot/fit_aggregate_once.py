"""One fixed aggregate-data continuation; ordinary final60000 only."""
import json
from pathlib import Path
import time
import numpy as np
import onnxruntime as ort
import torch

from fit_linear_head import CONFIG, make_actor, export
from continue_linear_fit import rng_save, rng_restore
from student_linear_runtime import BASE, KIND, FROZEN_RECEIPT, assert_frozen, archive, sha

OLD = BASE.parent / 'fast_controller_nominal_pilot_v1'
PRIOR = BASE.parent / 'fast_controller_continued_fit_v1'
FRESH = BASE.parent / 'fresh_expert_labels_resume_v1'
DEST = BASE / 'fit'
FIRST, LAST = 20001, 60000
PHASES = ('entry', 'acquisition', 'source', 'return')


def write(name, value):
    (DEST / name).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def same_tree(left, right):
    if torch.is_tensor(left):
        return torch.is_tensor(right) and torch.equal(left, right)
    if isinstance(left, dict):
        return left.keys() == right.keys() and all(same_tree(left[k], right[k]) for k in left)
    if isinstance(left, (list, tuple)):
        return len(left) == len(right) and all(same_tree(x, y) for x, y in zip(left, right))
    return left == right


def phase_indices(control):
    return [np.flatnonzero(mask) for mask in (control < 250,
        (control >= 250) & (control < 350), (control >= 350) & (control < 1169), control >= 1169)]


def summarize(error, residual_error, indices):
    e, r = error[indices], residual_error[indices]
    assert len(e)
    return dict(rows=len(e), applied_target_rmse_rad=float(np.sqrt(np.mean(e**2))),
        applied_target_rmse_by_joint_rad=np.sqrt(np.mean(e**2, axis=0)).tolist(),
        applied_target_abs_error_p95_by_joint_rad=np.percentile(np.abs(e), 95, axis=0).tolist(),
        applied_target_max_error_rad=float(np.max(np.abs(e))),
        preclip_residual_rmse_rad=float(np.sqrt(np.mean(r**2))))


def diagnostics(actor, features, span, data, datasets, controls, strata, pairs, step):
    with torch.inference_mode():
        predicted = (actor(features) * torch.from_numpy(span)).numpy()
    target = np.clip(data['base_target'] + predicted, data['joint_limits'][:, 0], data['joint_limits'][:, 1])
    error, residual_error = target-data['expert_target'], predicted-data['residual_rad']
    assert np.isfinite(predicted).all() and np.isfinite(error).all()
    summary = {'all': summarize(error, residual_error, np.arange(len(error)))}
    for dataset, code in [('old', 0), ('new', 1)]:
        indices = np.flatnonzero(datasets == code)
        summary[dataset] = dict(all=summarize(error, residual_error, indices),
            phases={name: summarize(error, residual_error, ids) for name, ids in zip(PHASES, strata[code])},
            first24_global_controls=summarize(error, residual_error, indices[controls[indices] < 24]),
            first24_dataset_rows=summarize(error, residual_error, indices[:24]))
    pair_rows = []
    for pair in pairs:
        old_i, new_i = pair['old_row'], 1269 + pair['new_row']
        pair_rows.append(dict(pair, predicted_target_difference_rad=(target[new_i]-target[old_i]).tolist(),
            actual_target_difference_rad=(data['expert_target'][new_i]-data['expert_target'][old_i]).tolist(),
            old_target_error_rad=error[old_i].tolist(), new_target_error_rad=error[new_i].tolist(),
            predicted_residual_difference_rad=(predicted[new_i]-predicted[old_i]).tolist(),
            actual_residual_difference_rad=(data['residual_rad'][new_i]-data['residual_rad'][old_i]).tolist()))
    query = 1269
    result = dict(global_step=step, additional_updates=max(0, step-20000), datasets=summary,
        actual_student_query_control1=dict(target_error_rad=error[query].tolist(),
            target_rmse_rad=float(np.sqrt(np.mean(error[query]**2))),
            residual_error_rad=residual_error[query].tolist()), fixed_pairs=pair_rows,
        fit_diagnostics_only=True, physical_evaluations=0, checkpoint_selection=False)
    return result, predicted, target, error


def main():
    assert_frozen()
    assert not (DEST / 'student_head.pt').exists() and not (DEST / 'report.json').exists()
    torch.set_num_threads(1)
    old, new = archive(OLD / 'labels/labels.npz'), archive(FRESH / 'labels/labels.npz')
    fresh_report = json.loads((FRESH / 'labels/report.json').read_text())
    assert fresh_report['samples'] == 1268 and sha(FRESH / 'labels/labels.npz') == fresh_report['labels_sha256']
    np.testing.assert_array_equal(old['control'], np.arange(1269))
    np.testing.assert_array_equal(new['control'], np.arange(1, 1269))
    np.testing.assert_array_equal(new['source_frame'], np.arange(12, 1280))
    saved = torch.load(PRIOR / 'fit/student_head.pt', map_location='cpu', weights_only=True)
    assert saved['completed_steps'] == 20000
    span, mean, std = [saved[k].numpy().copy() for k in ('joint_span', 'feature_mean', 'feature_std')]
    for data in (old, new):
        np.testing.assert_array_equal(data['joint_span'], span)
        np.testing.assert_array_equal(data['joint_limits'], old['joint_limits'])
    old_fit = archive(PRIOR / 'fit/teacher_fit.npz')
    np.testing.assert_array_equal(mean, old_fit['feature_mean'])
    np.testing.assert_array_equal(std, old_fit['feature_std'])
    data = {k: np.concatenate((old[k], new[k])) for k in
            ('features', 'residual_rad', 'base_target', 'expert_target', 'control', 'source_frame')}
    data['joint_limits'] = old['joint_limits'].copy()
    X = data['features'].astype(np.float32)
    features = torch.from_numpy((X-mean)/std)
    labels = torch.from_numpy((data['residual_rad']/span).astype(np.float32))
    datasets = np.r_[np.zeros(1269, np.int64), np.ones(1268, np.int64)]
    strata = [phase_indices(old['control']), [ids+1269 for ids in phase_indices(new['control'])]]
    assert [[len(ids) for ids in group] for group in strata] == [[250, 100, 819, 100], [249, 100, 819, 100]]
    actor = make_actor()
    optimizer = torch.optim.AdamW(actor.parameters(), lr=CONFIG['learning_rate'], weight_decay=CONFIG['weight_decay'])
    actor.load_state_dict(saved['actor_state'])
    optimizer.load_state_dict(saved['optimizer_state'])
    rng_restore(saved['rng'])
    assert same_tree(actor.state_dict(), saved['actor_state'])
    assert same_tree(optimizer.state_dict(), saved['optimizer_state'])
    assert same_tree(rng_save(), saved['rng'])
    assert all(int(state['step']) == 20000 for state in optimizer.state.values())
    for group in optimizer.param_groups:
        assert group['lr'] == 3e-4 and group['weight_decay'] == 1e-5
    with torch.inference_mode():
        previous_prediction = (actor(features[:1269])*torch.from_numpy(span)).numpy()
    np.testing.assert_array_equal(previous_prediction, old_fit['predicted_delta'])
    export(actor, mean, std, span, DEST / 'restored20000.onnx')
    assert sha(DEST / 'restored20000.onnx') == sha(PRIOR / 'fit/student_head.onnx')
    assert same_tree(rng_save(), saved['rng'])
    compatibility = json.loads((FRESH / 'compatibility/report.json').read_text())
    near = archive(FRESH / 'compatibility/cross_distances_and_pairs.npz')
    pairs = []
    for name in ('nearest', 'nearest_outside_plusminus4'):
        selected = compatibility['nearest_neighbor'][name]
        rows = np.unique([0] + selected['twenty_closest_new_row_indices'] + selected['twenty_largest_finite_sensitivity_new_row_indices'])
        for row in rows:
            pairs.append(dict(selection=name, new_row=int(row), old_row=int(near[name+'_old_row'][row]),
                new_control=int(new['control'][row]), old_control=int(near[name+'_old_control'][row]),
                normalized_feature_rms_distance=float(near[name+'_feature_rms'][row])))
    request = dict(kind=KIND, experiment='one_qualified_actual_state_aggregate_continuation',
        first_global_step=FIRST, last_global_step=LAST, additional_updates=40000,
        samples=dict(old=1269, new=1268, aggregate=2537), batch_size=256,
        phase_rows=[[len(ids) for ids in group] for group in strata],
        sampling='for each phase,32 old then32 new rows with replacement, continuing saved NumPy RNG',
        optimizer=CONFIG, original_normalization_refitted=False, full_final20000_model_optimizer_rng_restored=True,
        prior_checkpoint_sha256=sha(PRIOR / 'fit/student_head.pt'),
        old_labels_sha256=sha(OLD / 'labels/labels.npz'), new_labels_sha256=sha(FRESH / 'labels/labels.npz'),
        fresh_labels_report_sha256=sha(FRESH / 'labels/report.json'),
        frozen_sources_sha256=sha(FROZEN_RECEIPT), ordinary_final_only=True,
        early_stopping=False, intermediate_physics_evaluations=False, new_expert_queries=False,
        canonical_rollout_requires_separate_final_export_review=True, hardware_authorized=False)
    write('request.json', request)
    write('restoration20000_parity.json', dict(model_tensors_exact=True, optimizer_tree_exact=True,
        rng_exact=True, normalization_span_exact=True, all1269_prior_predictions_exact=True,
        restored_onnx_bytes_exact=True, optimizer_parameter_states=len(optimizer.state),
        optimizer_steps=20000, updates_executed=0, prior_checkpoint_sha256=request['prior_checkpoint_sha256']))
    write('fixed_pairs.json', pairs)
    records = []
    before, _, _, _ = diagnostics(actor, features, span, data, datasets, data['control'], strata, pairs, 20000)
    history = [before]
    assert same_tree(rng_save(), saved['rng'])
    started = time.perf_counter()
    for step in range(FIRST, LAST+1):
        picked = np.concatenate([np.random.choice(strata[dataset][phase], 32, replace=True)
                                 for phase in range(4) for dataset in (0, 1)])
        prediction = actor(features[picked])
        loss = torch.mean((prediction-labels[picked])**2)
        if not torch.isfinite(loss):
            raise ValueError('nonfinite training loss at step%d' % step)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(actor.parameters(), CONFIG['gradient_clip_norm'], error_if_nonfinite=True)
        optimizer.step()
        if step % 100 == 0:
            records.append(dict(global_step=step, normalized_residual_training_loss=float(loss.detach()),
                                elapsed_seconds=time.perf_counter()-started))
        if step % 1000 == 0:
            diagnostic, predicted, targets, errors = diagnostics(actor, features, span, data, datasets,
                                                               data['control'], strata, pairs, step)
            diagnostic['elapsed_seconds'] = time.perf_counter()-started
            history.append(diagnostic)
            write('training_log.json', records)
            write('full_fit_metrics.json', history)
            print(json.dumps(dict(global_step=step,
                old_rmse=diagnostic['datasets']['old']['all']['applied_target_rmse_rad'],
                new_rmse=diagnostic['datasets']['new']['all']['applied_target_rmse_rad'],
                query1_rmse=diagnostic['actual_student_query_control1']['target_rmse_rad'],
                elapsed_seconds=diagnostic['elapsed_seconds'])), flush=True)
    assert step == LAST
    actor.eval()
    export(actor, mean, std, span, DEST / 'student_head.onnx')
    settings = ort.SessionOptions()
    settings.intra_op_num_threads = settings.inter_op_num_threads = 1
    settings.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    session = ort.InferenceSession(str(DEST / 'student_head.onnx'), sess_options=settings, providers=['CPUExecutionProvider'])
    exported = session.run(None, {'features': X})[0]
    maximum_difference = float(np.max(np.abs(exported-predicted)))
    assert maximum_difference < 1e-5, maximum_difference
    checkpoint = dict(kind=KIND, actor_state=actor.state_dict(), optimizer_state=optimizer.state_dict(),
        rng=rng_save(), feature_mean=torch.from_numpy(mean), feature_std=torch.from_numpy(std),
        joint_span=torch.from_numpy(span), completed_steps=60000, additional_updates=40000, request=request)
    torch.save(checkpoint, DEST / 'student_head.pt')
    np.savez_compressed(DEST / 'teacher_fit.npz', predicted_delta=predicted,
        predicted_applied_target=targets, expert_target=data['expert_target'], applied_error=errors,
        feature_mean=mean, feature_std=std, dataset=datasets, control=data['control'], source_frame=data['source_frame'])
    write('report.json', dict(kind=KIND, experiment=request['experiment'], steps=60000, additional_updates=40000,
        stop_reason='fixed40000 additional updates completed; ordinaryfinal60000', network_training_complete=True,
        ONNX_vs_Torch_max_delta_rad=maximum_difference, final_fit_diagnostic=diagnostic,
        teacher_state_training_metrics=diagnostic['datasets'], elapsed_seconds=time.perf_counter()-started,
        checkpoints={p.name: sha(p) for p in (DEST/'zero_head.onnx', DEST/'student_head.onnx', DEST/'student_head.pt')},
        restoration_parity_sha256=sha(DEST/'restoration20000_parity.json'),
        closed_loop_assessment_pending=True, no_generalization_claim=True, physical_evaluations=0,
        expert_queries=0, hardware_authorized=False))
    print(json.dumps(dict(training_complete=True, ordinary_final_global_step=60000,
          onnx_max_difference=maximum_difference, head_sha256=sha(DEST/'student_head.onnx'))), flush=True)


if __name__ == '__main__':
    main()
