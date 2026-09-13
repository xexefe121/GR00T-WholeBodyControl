"""Review saved initial/final diagnostics only; no model, inference or dynamics."""
import hashlib
import json
from pathlib import Path
import numpy as np

NEW = Path('E:/codex-artifacts/sonic23_teleop_resume_20260911')
BASE = NEW / 'velocity_chord_student_v1'
FIT = BASE / 'fit'
OUT = Path(__file__).parent


def sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def arrays(path):
    with np.load(path, allow_pickle=False) as source:
        return {key: source[key].copy() for key in source.files}


checks = 0
def eq(actual, expected, label):
    global checks
    a, b = np.asarray(actual), np.asarray(expected)
    assert a.shape == b.shape and np.array_equal(a, b), label
    checks += 1


def close(actual, expected, label):
    global checks
    assert np.isclose(actual, expected, rtol=2e-12, atol=2e-12), label
    checks += 1


def main():
    assert np.__version__ == '1.23.5'
    assert not (OUT / 'saved_diagnostics.json').exists()
    paths = list(FIT.glob('*'))
    paths += [BASE / 'generation/centers.npz', BASE / 'generation/base_target.npy',
              BASE / 'generation/teacher_target.npy', BASE / 'training_frozen_inputs.json',
              BASE / 'training_clearance.json', BASE / 'fit_process_status.json',
              NEW / 'bfm_entry250_labels_v1/compatibility/fixed_pairs.json',
              NEW / 'fast_controller_phase_fit_v1/nominal/trace.npz',
              Path('Z:/codex/GR00T-WholeBodyControl-sonic-transfer-23dof/artifacts/teleop_six_hour_20260910/mjbatch_native23_inputs_v1/contract.json')]
    pins = {str(p): sha(p) for p in paths if p.is_file()}
    request = dict(kind='saved_diagnostic_reconstruction_no_model_evaluation', input_hashes=pins,
                   source_sha256=sha(__file__), inference_calls=0, physics_steps=0, optimizer_steps=0)
    (OUT / 'request.json').write_text(json.dumps(request, indent=2) + '\n')
    c = arrays(BASE / 'generation/centers.npz')
    pb = np.load(BASE / 'generation/base_target.npy', mmap_mode='r')
    pt = np.load(BASE / 'generation/teacher_target.npy', mmap_mode='r')
    span = c['joint_span']; lo, hi = c['joint_limits'].T
    pairs = read(NEW / 'bfm_entry250_labels_v1/compatibility/fixed_pairs.json')
    contract = read(paths[-1])
    trace = arrays(NEW / 'fast_controller_phase_fit_v1/nominal/trace.npz')
    groups = [[np.flatnonzero((c['dataset'] == d) & (c['control'] >= a) & (c['control'] < b))
               for a, b in ((250, 350), (350, 1169), (1169, 1269))] for d in range(3)]
    eq([[len(x) for x in group] for group in groups], [[100, 819, 100]] * 3, 'strata')
    stage_summary = {}
    for stage, step in (('initial', 65000), ('final', 70000)):
        predicted = arrays(FIT / (stage + '_predictions.npz'))
        nominal = read(FIT / (stage + '_nominal_metrics.json'))
        chords = read(FIT / (stage + '_chord_metrics.json'))
        sensitivity = read(FIT / (stage + '_sensitivity.json'))
        jacobians = arrays(FIT / (stage + '_jacobians.npz'))
        delta = predicted['predicted_delta']; proposed = c['base_target'] + delta[:3057]
        applied = np.clip(proposed, lo, hi); error = applied - c['expert_target']
        raw_error = delta[:3057] - c['residual_rad']
        def check_nominal(record, ids, name):
            eq(record['rows'], len(ids), name + '/rows')
            eq(record['applied_target_rmse_rad'], np.sqrt(np.mean(error[ids] ** 2)), name + '/RMS')
            eq(record['applied_target_by_joint_rmse_rad'], np.sqrt(np.mean(error[ids] ** 2, axis=0)), name + '/joint')
            eq(record['unclipped_target_rmse_rad'], np.sqrt(np.mean(raw_error[ids] ** 2)), name + '/raw')
            eq(record['clipped_rows'], np.any(proposed[ids] != applied[ids], axis=-1).sum(), name + '/clip')
        eq(nominal['global_step'], step, stage + '/step')
        check_nominal(nominal['all'], np.arange(3057), stage + '/all')
        for d, group in enumerate(groups):
            record = nominal['datasets'][d]
            eq(record['dataset'], d, stage + '/dataset')
            check_nominal(record['all'], np.concatenate(group), stage + '/dataset_all')
            for phase, ids in enumerate(group):
                check_nominal(record['phases'][phase], ids, stage + '/phase')
                check_nominal(record['first24_each_phase'][phase], ids[:24], stage + '/first24')
        eq(nominal['query250_signed_error_rad'], error[2038], stage + '/query250_signed')
        eq(nominal['query250_target_rmse_rad'], np.sqrt(np.mean(error[2038] ** 2)), stage + '/query250_RMS')
        eq(len(nominal['fixed_pairs']), len(pairs), stage + '/paircount')
        for saved, pair in zip(nominal['fixed_pairs'], pairs):
            for key, value in pair.items():
                assert saved[key] == value, (stage, key)
            i, j = pair['a_global_row'], pair['b_global_row']
            values = dict(predicted_target_difference_rad=applied[j]-applied[i],
                actual_target_difference_rad=c['expert_target'][j]-c['expert_target'][i],
                a_target_error_rad=error[i], b_target_error_rad=error[j],
                predicted_residual_difference_rad=delta[j]-delta[i],
                actual_residual_difference_rad=c['residual_rad'][j]-c['residual_rad'][i])
            for key, value in values.items(): eq(saved[key], value, stage + '/' + key)
        probe_u = pb + delta[3057:].reshape(3057, 23, 2, 23)
        teacher_change = pt - c['expert_target'][:, None, None, :]
        residual = probe_u - proposed[:, None, None, :] - teacher_change
        normalized = (residual / span) ** 2
        probe_a = np.clip(probe_u, lo, hi)
        applied_residual = probe_a - applied[:, None, None, :] - teacher_change
        for d, group in enumerate(groups):
            for phase, ids in enumerate(group):
                cell = chords['cells'][d*3+phase]
                for record, indices in ((cell, ids), (cell['first24'], ids[:24])):
                    eq(record['rows'], len(indices), stage + '/chord_rows')
                    eq(record['axis_pairs'], len(indices)*23, stage + '/chord_pairs')
                    eq(record['normalized_proposal_chord_MSE'], normalized[indices].mean(), stage + '/chord_MSE')
                    eq(record['proposal_chord_error_rms_rad'], np.sqrt(np.mean(residual[indices]**2)), stage + '/chord_RMS')
                    eq(record['applied_chord_error_rms_rad'], np.sqrt(np.mean(applied_residual[indices]**2)), stage + '/applied_chord_RMS')
                    eq(record['student_clipped_probes'], np.any(probe_u[indices] != probe_a[indices], axis=-1).sum(), stage + '/chord_clip')
                eq(cell['student_clipped_components'], np.sum(probe_u[ids] != probe_a[ids]), stage + '/chord_component_clip')
        blocks = dict(joint_position=(0,23),joint_velocity=(23,46),root_angular_velocity=(46,49),gravity=(49,52),
                      previous_target=(52,75),root_linear_velocity=(75,78),root_height=(78,79),received_goals=(79,1023),
                      BFM_base=(1023,1046),previous_raw_action=(1046,1069))
        norm = lambda x: float(np.linalg.svd(x, compute_uv=False)[0])
        scale = .25*np.asarray(contract['training_effort'])/np.asarray(contract['kp'])
        inv = 1/scale; limits = np.asarray(contract['joint_limits'])
        for row in sensitivity['rows']:
            control = row['control']; prefix = 'control' + str(control)
            J = jacobians[prefix + '_J_raw_features']
            assert J.shape == (23,1069) and np.isfinite(J).all()
            prior = trace['previous_action'][control].astype(np.float64)
            prior_target = np.asarray(contract['default_q']) + prior*scale
            mask = ((prior_target > limits[:,0]) & (prior_target < limits[:,1])).astype(float)
            R = J[:,52:75]*mask[None,:] + J[:,1046:1069]*inv[None,:]
            Raw = inv[:,None]*R*scale[None,:]; B = np.eye(23)+J[:,1023:1046]
            for key, value in (('_conditional_prior_target',R),('_conditional_prior_raw',Raw),('_conditional_BFM_base',B)):
                eq(jacobians[prefix+key],value,stage+key)
            for name,(a,b) in blocks.items():close(row['feature_block_spectral_norm'][name],norm(J[:,a:b]),stage+'/'+name)
            for key,value in dict(conditional_prior_target_equivalent_norm=norm(R),conditional_prior_raw_action_norm=norm(Raw),
                conditional_prior_spectral_radius=float(np.max(np.abs(np.linalg.eigvals(R)))),conditional_BFM_base_total_target_norm=norm(B)).items():
                close(row[key],value,stage+'/'+key)
            eq(row['previous_target_clipped_inputs'],np.sum(mask==0),stage+'/prior_mask')
        stage_summary[stage] = dict(nominal_all_RMS=nominal['all']['applied_target_rmse_rad'],query250_RMS=nominal['query250_target_rmse_rad'],
            dataset_RMS=[d['all']['applied_target_rmse_rad'] for d in nominal['datasets']],
            full_chord_objective=chords['full_nine_cell_chord_objective'],conditional_velocity_norm_at250=sensitivity['rows'][0]['feature_block_spectral_norm']['joint_velocity'])
    for path, digest in pins.items():assert sha(path)==digest,path
    result = dict(pass_all=True,checks=checks,exact_arithmetic_checks=True,spectral_summary_tolerance=2e-12,
        input_hashes=pins,request_sha256=sha(OUT/'request.json'),source_sha256=sha(__file__),stages=stage_summary,
        scope=['All saved nominal per-joint/phase/first24/query250 and 243 fixed-pair diagnostics for both heads.',
               'All saved chord phase and first24 summaries including clipping for both heads.',
               'Saved Jacobian conditional maps and spectral summaries only; no new head/Jacobian evaluation.'],
        excluded=['Nominal scalar objective and optimizer/export/RNG checks are covered by root saved-fit audit.',
                  'No independent authenticity assertion for saved Jacobians or their float64-vs-Torch evaluation error.'],
        inference_calls=0,physics_steps=0,optimizer_steps=0)
    (OUT/'saved_diagnostics.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(pass_all=True,checks=checks,stages=stage_summary,sha256=sha(OUT/'saved_diagnostics.json'))))


if __name__ == '__main__': main()
