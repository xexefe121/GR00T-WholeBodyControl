"""Saved-array feasibility for nominal final70000 one-control policy branches."""
from pathlib import Path
import hashlib
import importlib.util
import json
import numpy as np

BASE=Path(__file__).parent;NEW=BASE.parent;OLD=NEW.parent/'sonic23_teleop_six_hour_20260910'
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def archive(path):
    with np.load(path,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}
def exact(a,b):
    a,b=np.asarray(a),np.asarray(b)
    return a.shape==b.shape and a.dtype==b.dtype and a.tobytes()==b.tobytes()


def main():
    source=NEW/'velocity_chord_student_v1/source_snapshot_v3'
    obs_path=source/'gear_sonic/utils/g1_true23_bfm_seed_observations.py'
    spec=importlib.util.spec_from_file_location('saved_policy_observations',obs_path)
    observations=importlib.util.module_from_spec(spec);spec.loader.exec_module(observations)
    centers_path=NEW/'velocity_chord_student_v1/generation/centers.npz'
    prediction_path=NEW/'velocity_chord_student_v1/fit/final_predictions.npz'
    witness_path=NEW/'velocity_chord_student_evaluation_v1/head_witness/witness.npz'
    actual_path=NEW/'velocity_chord_student_evaluation_v1/nominal/trace.npz'
    contract_path=Path('Z:/codex/GR00T-WholeBodyControl-sonic-transfer-23dof/artifacts/teleop_six_hour_20260910/mjbatch_native23_inputs_v1/contract.json')
    centers=archive(centers_path);predictions=archive(prediction_path);witness=archive(witness_path);actual=archive(actual_path);contract=read(contract_path)
    selected=np.flatnonzero(centers['control']<1268);assert len(selected)==3054
    assert predictions['onnx_predicted_delta'].shape==(143679,23)
    c={k:np.asarray(contract[k]) for k in ('default_q','kp','training_effort','joint_limits')}
    lo,hi=c['joint_limits'].T;delta=predictions['onnx_predicted_delta'][selected]
    proposal=centers['base_target'][selected]+delta;target=np.clip(proposal,lo,hi)
    raw_prior=(centers['base_action'][selected]+delta*c['kp']/(.25*c['training_effort'])).astype(np.float32)
    applied_prior=((target-c['default_q'])*c['kp']/(.25*c['training_effort'])).astype(np.float32)
    checks=[]
    def check(name,a,b):
        value=exact(a,b);checks.append(dict(name=name,byte_exact=value));assert value,name
    for dataset in range(3):
        rows=selected[centers['dataset'][selected]==dataset]
        check(f'dataset{dataset} controls',centers['control'][rows],np.arange(250,1268,dtype=np.int64))
        check(f'dataset{dataset} next controls',centers['control'][rows+1],np.arange(251,1269,dtype=np.int64))
        check(f'dataset{dataset} same next dataset',centers['dataset'][rows+1],np.full(1018,dataset,np.int64))
    row=2038;at=int(np.flatnonzero(selected==row)[0])
    check('query250 original nominal features',centers['features'][row],actual['features'][250])
    check('query250 Windows batch delta equals actual WSL delta',predictions['onnx_predicted_delta'][row],actual['delta'][250])
    check('query250 Windows batch delta equals separate WSL witness',predictions['onnx_predicted_delta'][row],witness['onnx_delta'])
    check('query250 proposed first target',target[at],actual['target'][250])
    check('query250 raw combined first prior',raw_prior[at],actual['action'][250])
    named={key:centers['history_'+key][row].copy() for key in ('actions','base_ang_vel','dof_pos','dof_vel','projected_gravity')}
    _,terms=observations.state_and_terms(centers['qpos'][row,7:],centers['qvel'][row,6:],centers['qpos'][row,3:7],
        centers['qvel'][row,3:6],centers['previous_action'][row],c['default_q'])
    for key in named:named[key][1:]=named[key][:-1].copy();named[key][0]=terms[key]
    check('query250 shifted history at actual251',np.concatenate([named[k].reshape(-1) for k in sorted(named)]),actual['control_history_before'][251])
    check('query250 raw prior at actual251',raw_prior[at],actual['control_previous_action_before'][251])
    baseline=archive(NEW/'bfm_entry250_actual_oracle_v1/nominal/trace.npz')
    check('query250 same full291 at starting250',baseline['control_integration_before'][250],actual['control_integration_before'][250])
    results=[]
    for dataset in range(3):
        at=np.flatnonzero(centers['dataset'][selected]==dataset)
        results.append(dict(dataset=dataset,rows=len(at),controls=[250,1267],label_controls=[251,1268],
            native_target_clipped_rows=int(np.count_nonzero(np.any(proposal[at]!=target[at],axis=1))),
            native_target_clipped_components=int(np.count_nonzero(proposal[at]!=target[at])),
            raw_prior_differs_from_applied_values_rows=int(np.count_nonzero(np.any(raw_prior[at]!=applied_prior[at],axis=1))),
            max_raw_combined_action_abs=float(np.max(np.abs(raw_prior[at]))),
            target_next_original_replan_centers=int(np.count_nonzero(centers['plan_local'][selected[at]+1]==0))))
    paths=[centers_path,prediction_path,witness_path,actual_path,contract_path,obs_path,
        source/'chord_fit_diagnostics.py',source/'fit_velocity_chords.py',source/'student_linear_runtime.py',
        NEW/'velocity_chord_student_v1/fit/report.json',NEW/'velocity_chord_final_export_review_v1/review.json',
        NEW/'velocity_fit_evidence_independent_v1/report.json',NEW/'velocity_fit_evidence_independent_v1/request.json',
        NEW/'velocity_chord_student_evaluation_v1/canonical_prefix250_parity.json',
        NEW/'velocity_chord_student_evaluation_v1/actual_query250_input_parity.json',
        BASE/'report.json']
    old_base=OLD/'bfm_online_intent_v2'
    old_reports={}
    for name in ('walk003_terminal_bfm_yaw4_hybrid_v1','walk003_quiet_frozen_import_verification_v3','walk003_allmargin_wsl_independent_replay_v1'):
        folder=old_base/name;path=folder/'report.json';report=read(path);paths.extend([path,folder/'trace.npz'])
        z=archive(folder/'trace.npz')
        old_reports[name]=dict(path=str(path),sha256=sha(path),trace_sha256=sha(folder/'trace.npz'),
            completed_controls=report['completed_controls'],physics_steps=report['physics_steps'],
            full291_initial_or_final_fields_present=[k for k in z if 'integration' in k or 'warmstart' in k],
            note='Producer/replay native report with recorded trace binding; not the new root capture report schema.')
    report=dict(kind='saved_nominal_final70000_one_control_behavior_branch_feasibility',assessment_complete=True,
        physics_steps=0,inference_calls=0,new_labels=0,fit=False,branches=3054,datasets=results,checks=checks,
        source_prediction_order='First3057 final_predictions.onnx_predicted_delta rows are centers.features in unchanged dataset/control order, followed by probes; frozen diagnostic source and root fit audit bound.',
        saved_prediction_backend='Windows ORT batch256; no general WSL batch1 byte-equivalence claim.',
        query250_first_saved_delta_matches_actual_and_WSL_witness_byte_exact=True,
        exact_actual251_reproduction_not_yet_executed=True,
        query250_actual251_comparison_fixture='Actual final70000 nominal trace full291/control_history_before/control_previous_action_before at251 plus native samples2500..2510. Same initial250 state, target and raw prior are already byte-exact; future dynamics must prove all10 samples and full291 endpoint.',
        raw_policy_semantics='Use original float32(base_action+delta*kp/(.25*training_effort)) at c+1, even when target clip differs. Advance H using nominal precontrol c terms and incoming prior. New raw policy prior enters separate last_action, not newest history entry.',
        applied_semantics='Inverse clipped target is a distinct optional controller convention. It does not preserve unchanged deployment raw prior when clipping or arithmetic differs; never substitute it silently.',
        nominal_zero_gate='Original actual expert command c through10 private steps must recover original c+1 full291/native samples/history/prior. Reuse the qualified saved nominal BFM center after pure exact input checks; no new nominal BFM call is included in the narrow budget.',
        future_maximum_counts=dict(nominal_native_steps=30540,policy_native_steps=30540,total_native_steps=61080,
            backward=3054,actor=3054,head_if_saved_commands_explicitly_selected=0,
            additional_WSL_batch1_head_if_true_deployment_commands_selected=3054,
            old_prefix_reconstruction_additional_steps=12680),
        failed_branch_policy='Stop each unsafe branch at its first strict2ms violation; preserve commanded clipping, full attempted inputs/native partial samples, requested row and failed validity. No label inference at invalid resulting state. Counts above are maxima, not guaranteed completed labels.',
        teacher='Original next-control c+1 plan/local, full58-column tangent K product and both clips, with no optimizer/replan. Applies only as frozen local teacher map, especially at next-control plan starts.',
        limitations=['Only one policy action from nominal teacher occupancy; it does not reproduce successive policy-history drift or recovery.',
            'Saved batch outputs are authentic final70000 nominal predictions but are not generally verified WSL batch1 commands. A separately selected3054-call WSL head stage would close this gap.',
            'The saved first query250 prediction happens to match the real WSL witness byte for byte; no guarantee extends to other rows.',
            'At c+1 a clipped physical target may coexist with a very different raw prior. Preserve both diagnostics and the explicitly selected raw convention.',
            'Teacher map validity and unsafe branch coverage must remain explicit; no full source/quiet/controller claim follows from these labels.'],
        old_existing_reports=old_reports,old_full291_endpoint_found=False,
        old_full291_search_scope='Old hybrid original, launcher replay and frozen-import v3 traces, their directory files, plus named endpoint/state NPZ inventory in NEW and OLD; none provides old full291 final endpoint.',
        old_capture_adaptation='Separate derived runner needed: bind original trace/report/initial canonical fixture; remove nonexistent endpoint/report-schema assumptions; compare warnings through explicit original int64 serialization while restoring native int32 zero ledgers. Do not fabricate endpoint proof.',
        input_sha256={str(p).replace('\\','/'):sha(p) for p in paths},source_sha256=sha(__file__))
    (BASE/'behavior_report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(branches=3054,checks_pass=len(checks),datasets=results,report_sha256=sha(BASE/'behavior_report.json'))))


if __name__=='__main__':main()
