"""Read saved arrays only: scalar design evidence, no labels/model/physics calls."""
from pathlib import Path
import hashlib
import json
import numpy as np

HERE = Path(__file__).resolve().parent
NEW = HERE.parent
CONTRACT = Path('Z:/codex/GR00T-WholeBodyControl-sonic-transfer-23dof/artifacts/teleop_six_hour_20260910/mjbatch_native23_inputs_v1/contract.json')
INPUTS = {}

def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def read(p):
    INPUTS[str(p)] = sha(p)
    return json.loads(Path(p).read_text(encoding='utf-8-sig'))

def load(p):
    INPUTS[str(p)] = sha(p)
    with np.load(p, allow_pickle=False) as z:
        return {k: z[k].copy() for k in z.files}

def stats(a):
    a = np.asarray(a, np.float64)
    return dict(min=float(a.min()), median=float(np.median(a)),
                p95=float(np.percentile(a, 95)), max=float(a.max()))

def main():
    assert not (HERE / 'scales.json').exists()
    c = read(CONTRACT)
    caps = np.asarray(c['native_velocity'], np.float64)
    a = load(NEW / 'phase_student_failure_diagnosis_v1/arrays.npz')
    failure = read(NEW / 'phase_student_failure_diagnosis_v1/report.json')
    sensitivity = read(NEW / 'phase_student_sensitivity_review_v1/report.json')
    feedback = read(NEW / 'three_expert_feedback_audit_v1/report.json')
    rows = read(NEW / 'three_expert_feedback_audit_v1/per_control.json')
    datasets = {}
    configs = [('old', 'fast_controller_nominal_pilot_v1'),
               ('query1', 'fresh_expert_labels_resume_v1'),
               ('query250', 'bfm_entry250_labels_v1')]
    for name, relative in configs:
        z = load(NEW / relative / 'labels/labels.npz')
        ids = np.flatnonzero((z['control'] >= 250) & (z['control'] < 1269))
        np.testing.assert_array_equal(z['control'][ids], np.arange(250, 1269))
        v = z['teacher_qvel'][:-1][ids, 6:]
        if name == 'query250':
            teacher_v = v
            span = z['joint_span'].astype(np.float64)
        r = [r for r in rows if r['dataset'] == name]
        energy = np.square([r['joint_velocity_K_frobenius'] for r in r])
        energy = np.sort(energy)[::-1]
        radii = [r['smallest_interior_single_joint_velocity_axis_radius_radps'] for r in r
                 if r['smallest_interior_single_joint_velocity_axis_radius_radps'] is not None]
        datasets[name] = dict(
            rows=len(ids), max_precontrol_native_speed_fraction=float(np.max(np.abs(v) / caps)),
            native_speed_margin_fraction=float(1-np.max(np.abs(v) / caps)),
            all_axis_plusminus_001_native_caps_statically_inside_native_speed_bounds=bool(np.all(np.abs(v) + .01*caps < caps)),
            joint_velocity_abs_radps=stats(np.abs(v)),
            next_expert_control_velocity_change_cap_normalized_rms=stats(np.sqrt(np.mean((np.diff(v, axis=0)/caps)**2, axis=1))),
            raw_velocity_K_squared_frobenius_energy_top1_fraction=float(energy[0]/energy.sum()),
            raw_velocity_K_squared_frobenius_energy_top10_fraction=float(energy[:10].sum()/energy.sum()),
            smooth_common_axis_radius_radps=stats(radii),
            rows_with_no_reported_smooth_common_axis_radius=len(r)-len(radii),
            saved_feedback_summary={k:feedback['datasets'][name][k] for k in [
                'joint_velocity_spectral_K','joint_position_spectral_K',
                'feedback_clipped_rows','native_target_clipped_rows',
                'velocity_derivative_nondifferentiable_controls','zero_gain_controls']})
    departures=[]
    for i, control in enumerate(a['control']):
        d = a['actual_precontrol_qvel'][i, 6:] - teacher_v[int(control)-250]
        z = d/caps
        departures.append(dict(control=int(control), joint_velocity_delta_rms_radps=float(np.sqrt(np.mean(d*d))),
            joint_velocity_delta_max_radps=float(np.abs(d).max()),
            delta_cap_normalized_rms=float(np.sqrt(np.mean(z*z))),
            delta_cap_normalized_max=float(np.abs(z).max())))
    result = dict(kind='saved_only_learner_neighborhood_scale_analysis', datasets=datasets,
        failure_velocity_departures=departures,
        selected_proposed_radius_native_speed_fraction=.01,
        selected_axis_step_radps=(.01*caps).tolist(),
        first251_delta_rms_relative_to_one_percent_caps=float(departures[1]['delta_cap_normalized_rms']/.01),
        axis_step_cap_normalized_rms=float(.01/np.sqrt(23)),
        teacher_center_to_perturbed_target_global_component_bound_rad=.2,
        bound_reason='Both exact fixed-plan feedback corrections belong to [-0.1,0.1]; outer native clipping is nonexpansive.',
        teacher_center_to_perturbed_target_bound_native_span_fraction=(.2/span).tolist(),
        all_3057_nominal_rows_retained=True,
        proposed_full_axis_pairs=3057*23, proposed_perturbed_inputs=3057*23*2,
        first_input_departure=failure['first_actual_input_differs_from_matching_expert'],
        first_target_clip=failure['first_native_target_clipping'],
        first_history_departure=failure['first_named_history_differs_from_matching_expert'],
        limitation='Fixed axial coordinate chords do not cover the simultaneous multi-joint control251 displacement or certify any perturbed trajectory.',
        model_inference_calls=0, optimizer_calls=0, physics_steps=0,
        generated_target_labels=0, generated_feature_rows=0)
    INPUTS[str(Path(__file__))] = sha(__file__)
    result['input_sha256'] = INPUTS
    (HERE/'scales.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    print(json.dumps(dict(scales_sha256=sha(HERE/'scales.json'), datasets={n:{k:v for k,v in d.items() if k not in ['saved_feedback_summary']} for n,d in datasets.items()}, first251=departures[1]), indent=2))

if __name__ == '__main__':
    main()
