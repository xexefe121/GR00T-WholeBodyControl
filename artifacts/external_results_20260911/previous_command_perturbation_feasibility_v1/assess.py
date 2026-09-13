"""Read-only source/data feasibility; no native model, dynamics or inference."""
from pathlib import Path
import hashlib
import importlib.util
import json
import sys
import numpy as np

BASE=Path(__file__).parent;NEW=BASE.parent;OLD=NEW.parent/'sonic23_teleop_six_hour_20260910'
SOURCE=NEW/'velocity_chord_student_v1/source_snapshot_v3'
ROOT=Path('Z:/codex/GR00T-WholeBodyControl-sonic-transfer-23dof')
sys.path.insert(0,str(SOURCE))
from chord_common import dataset_configs
from saved_committed_map import CORE,difference_function


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def archive(p):
    with np.load(p,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}
def exact(a,b):
    a,b=np.asarray(a),np.asarray(b)
    return a.shape==b.shape and a.dtype==b.dtype and a.tobytes()==b.tobytes()


def main():
    contract_path=ROOT/'artifacts/teleop_six_hour_20260910/mjbatch_native23_inputs_v1/contract.json'
    centers_path=NEW/'velocity_chord_student_v1/generation/centers.npz'
    observation_path=SOURCE/'gear_sonic/utils/g1_true23_bfm_seed_observations.py'
    module_spec=importlib.util.spec_from_file_location('pure_observations',observation_path)
    observations=importlib.util.module_from_spec(module_spec);module_spec.loader.exec_module(observations)
    contract=read(contract_path);centers=archive(centers_path);difference=difference_function()
    limits=np.asarray(contract['joint_limits']);kp=np.asarray(contract['kp']);effort=np.asarray(contract['training_effort']);default=np.asarray(contract['default_q'])
    keys=('actions','base_ang_vel','dof_pos','dof_vel','projected_gravity')
    widths=(23,3,23,23,3)
    comparisons=[];input_paths=[contract_path,centers_path,observation_path,CORE,SOURCE/'chord_common.py',SOURCE/'saved_committed_map.py',
        SOURCE/'gear_sonic/utils/g1_true23_mjbatch_bfm_seed.py',SOURCE/'student_linear_runtime.py',
        NEW/'velocity_chord_student_v1/generation_frozen_inputs_v2.json',
        NEW/'velocity_chord_completed_data_review_v1/review.json',
        NEW/'expert_resumed_root_qualification_v1/qualification.json',NEW/'bfm250_expert_root_qualification_v1/qualification.json',
        NEW/'velocity_chord_student_v1/generation/report.json']
    def check(name,a,b):
        value=exact(a,b);comparisons.append(dict(name=name,byte_exact=value))
        assert value,name
    datasets=[]
    for dataset,config in enumerate(dataset_configs()):
        input_paths.extend([config['trace'],config['labels']])
        trace=archive(config['trace']);labels=archive(config['labels'])
        center_ids=np.flatnonzero(centers['dataset']==dataset)
        check(config['name']+' selected controls',centers['control'][center_ids],np.arange(250,1269,dtype=np.int64))
        named={key:np.zeros((4,width),np.float32) for key,width in zip(keys,widths)}
        prior=np.zeros(23,np.float32);before={}
        source_trace_history='history' if dataset==0 else 'control_history_before'
        source_trace_prior='previous_action' if dataset==0 else 'control_previous_action_before'
        for control in range(1269):
            history=np.concatenate([named[key].reshape(-1) for key in keys]).copy()
            check(config['name']+f' history {control}',history,trace[source_trace_history][control])
            check(config['name']+f' prior {control}',prior,trace[source_trace_prior][control])
            if control>=249:before[control]=(history.copy(),prior.copy())
            state,terms=observations.state_and_terms(trace['qpos'][control,7:],trace['qvel'][control,6:],
                trace['qpos'][control,3:7],trace['qvel'][control,3:6],prior,default)
            for key in keys:
                named[key][1:]=named[key][:-1].copy();named[key][0]=terms[key]
            use_raw=(dataset==1 and control==0) or (dataset==2 and control<250)
            prior=(trace['action'][control].copy() if use_raw else
                ((trace['target'][control]-default)*kp/(.25*effort)).astype(np.float32))
        for row in center_ids:
            control=int(centers['control'][row]);previous_control=control-1
            previous_history,previous_prior=before[previous_control]
            current_history,current_prior=before[control]
            check(config['name']+f' center history {control}',current_history,centers['history'][row])
            check(config['name']+f' center prior {control}',current_prior,centers['previous_action'][row])
            # Zero perturbation of the actually applied native target exactly
            # recreates the target-to-prior convention, including BFM c249.
            inferred=((trace['target'][previous_control]-default)*kp/(.25*effort)).astype(np.float32)
            check(config['name']+f' predecessor actual target prior {control}',inferred,current_prior)
            state=np.r_[trace['qpos'][control],trace['qvel'][control]]
            check(config['name']+f' center qpos {control}',state[:30],centers['qpos'][row])
            check(config['name']+f' center qvel {control}',state[30:],centers['qvel'][row])
            raw=centers['gain'][row]@difference(centers['planned_state'][row],state)
            correction=np.clip(raw,-.1,.1)
            target=np.clip(centers['planned_target'][row]+correction,limits[:,0],limits[:,1])
            check(config['name']+f' teacher raw {control}',raw,centers['teacher_feedback_raw'][row])
            check(config['name']+f' teacher target {control}',target,trace['target'][control])
        coverage=dict(dataset=dataset,name=config['name'],centers=1019,center_controls=[250,1268],
            predecessor_controls=[249,1267],next_comparison_states=[250,1268],
            source_trace_path=str(config['trace']),source_trace_sha256=sha(config['trace']),
            history_prior_all_controls_byte_exact=1269,
            full291_precontrol_available='control_integration_before' in trace,
            zero_native_target_perturbation_prior_byte_exact_all1019=True,
            teacher_nominal_full_product_both_clips_byte_exact_all1019=True,
            center_plan_start_count=int(np.count_nonzero(centers['plan_local'][center_ids]==0)),
            target_exact_native_bound_components=int(np.count_nonzero(
                (trace['target'][249:1268]==limits[:,0])|(trace['target'][249:1268]==limits[:,1]))))
        warning_key='physics_warning_counts' if dataset==0 else 'physics_warning_number'
        check(config['name']+' all predecessor/center warning counts zero',trace[warning_key][2490:12690:10],np.zeros((1020,8),trace[warning_key].dtype))
        check(config['name']+' all predecessor/center warning lastinfo zero',trace['physics_warning_lastinfo'][2490:12690:10],np.zeros((1020,8),trace['physics_warning_lastinfo'].dtype))
        coverage['source_warning_dtype']=str(trace[warning_key].dtype)
        if coverage['full291_precontrol_available']:
            assert trace['control_integration_before'].shape==(1569,291)
            assert int(trace['integration_state_spec'])==8191
            for control in range(249,1269):
                vector=trace['control_integration_before'][control]
                check(config['name']+f' full291 q {control}',vector[1:31],trace['qpos'][control])
                check(config['name']+f' full291 dq {control}',vector[31:60],trace['qvel'][control])
                check(config['name']+f' full291 time {control}',vector[0],trace['physics_time'][control*10])
                check(config['name']+f' full291 preceding ctrl {control}',vector[89:112],trace['physics_torque'][control*10-1])
                check(config['name']+f' application forces zero {control}',vector[112:],np.zeros(179))
            coverage.update(full291_rows_before_and_after=1020,integration_spec=8191,
                qacc_warmstart_present=True,external_force_channels_all_zero=True,
                warnings_precontrol_shape=list(trace['physics_warning_number'][2490:12690:10].shape))
        else:
            coverage.update(missing_full291_rows=1020,
                required_future_work='Independent immutable recorded-command reconstruction of original trace through control1267 (12680 native steps) captures precontrols249..1268; or capture full1269 prefix including its endpoint with12690 steps. Not authorized by this assessment.',
                cannot_reconstruct_full291_from_qpos_qvel_alone=True)
        datasets.append(coverage)
    baseline_path=NEW/'original_bfm_entry250_v1/entry250/trace.npz';input_paths.append(baseline_path)
    baseline=archive(baseline_path);query=archive(dataset_configs()[2]['trace'])
    for key in ('control_integration_before','control_history_before','control_previous_action_before'):
        check('query250 c249 baseline '+key,query[key][249],baseline[key][249])
    check('query250 c250 full boundary',query['control_integration_before'][250],baseline['final_integration'])
    measured=read(NEW/'velocity_chord_student_v1/generation/report.json')
    result=dict(kind='previous_native_command_perturbation_read_only_feasibility',assessment_complete=True,
        physics_steps=0,BFM_calls=0,head_calls=0,new_labels=0,fitting=False,
        scope='Existing three walk003 expert datasets only; held-out walk008 and broader PICO/walk002 absent.',
        centers=3057,datasets=datasets,pure_byte_comparisons=len(comparisons),all_comparisons_pass=True,
        query250_c249_full291_history_prior_matches_qualified_BFM_prefix=True,
        query250_c250_full291_matches_BFM250_endpoint=True,
        one_control_history_semantics='At c-1, shift terms made from original precontrol q/dq/prior. The perturbed outgoing command enters the separate prior at c; it does not replace the newest history action, which is the prior entering c-1. Thus H_c remains exactly nominal for a single outgoing-command perturbation.',
        prior_semantics='Use inverse actually clipped perturbed native target in float32 without +/-5 clipping, as record_control does. Zero perturbation recovers every existing center prior byte-exact, including query250 c249 raw-BFM boundary.',
        BFM_semantics='At changed q/dq/root pose at c, rerun original yaw2 _goal at c+11 then actor and original1069 builder with shifted H_c/new prior. Original cached latent is not generally valid when root pose changes.',
        teacher_semantics='Use original plan/local at c. Full tangent difference, full58-column K product, clip correction to[-.1,.1], add original planned native target, clip native limits; no refit/replan or reduced-column shortcut.',
        nominal_replay_required='Before any probe for a row, restore exact full291, warning number/lastinfo and independent time at c-1. Original actual target through10 manual-PD native steps must match all saved q/dq/command/actualforce/time/warnings and full291 at c; q/dq-only reset is insufficient. Then original BFM/base/features/teacher target must match the center. Preserve first mismatches before probes.',
        branch_rules='Each probe starts from an isolated copy of the identical predecessor state/history/prior. Apply fixed bounded target for10 actual2ms steps, with original torque clip and strict oracle every step; never concatenate probes. Preserve invalid/rejected probes in requested denominator; no adaptive retries or omission.',
        cost_scenarios={str(directions):dict(probes=3057*directions,nominal_replay_steps=30570,
            probe_native_steps=30570*directions,nominal_plus_probe_actor_calls=3057*(directions+1),
            nominal_plus_probe_backward_calls=3057*(directions+1)) for directions in (2,23,46)},
        scale_not_selected=True,amplitude_not_selected=True,
        full46_schedule_minimum_zero_displacement_native_clipped_probes=sum(d['target_exact_native_bound_components'] for d in datasets),
        teacher_replan_boundary_centers=sum(d['center_plan_start_count'] for d in datasets),
        measured_reference_generation=dict(actor_calls=measured['inference_calls']['actor'],backward_calls=measured['inference_calls']['backward'],
            actor_seconds=measured['inference_seconds']['actor'],backward_seconds=measured['inference_seconds']['backward'],elapsed_seconds=measured['elapsed_seconds']),
        limitations=['Physics consistency covers one20ms command, not multistep deployment occupancy or source/quiet performance.',
            'At original replan boundaries, the original K/plan is a frozen local teacher map, not the optimizer result that a changed state would have produced.',
            'Native target clipping can collapse opposite/one-sided probes; record actual displacement, clip masks, duplicates and invalid outcomes.',
            'New actual-target prior follows teacher record_control. The unchanged student raw combined-action history can still disagree after deployment target clipping; this data alone does not fix that runtime mismatch.',
            'A bounded joint-target displacement does not guarantee contact/velocity safety; strict per2ms rejection remains necessary.',
            'The original dataset lacks full291 snapshots; reconstruction is a separate required authorized native pass before generation.',
            'No amplitude, direction set, generation, optimizer or evaluation is selected here.'],
        input_sha256={str(p).replace('\\','/'):sha(p) for p in input_paths},assessment_source_sha256=sha(__file__))
    (BASE/'comparisons.json').write_text(json.dumps(comparisons,indent=2)+'\n')
    result['comparison_sha256']=sha(BASE/'comparisons.json')
    (BASE/'report.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('input_sha256','limitations','nominal_replay_required','branch_rules','teacher_semantics','BFM_semantics','prior_semantics','one_control_history_semantics')}))


if __name__=='__main__':main()
