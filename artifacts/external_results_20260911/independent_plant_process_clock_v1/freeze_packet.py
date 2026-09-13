"""Concrete saved-input/source freeze only. Does not select or execute a run."""
import json,sys
from pathlib import Path
import numpy as np
BASE=Path(__file__).resolve().parent;NEW=BASE.parent;SOURCE=BASE/'source_runner_v1'
sys.path.insert(0,str(SOURCE))
from packet_io import sha,read,write,require_roles
from recorded_protocol import canonical_table


def main():
    request_path=BASE/'clock_request.json'
    if request_path.exists():raise RuntimeError('preserve existing concrete request; no overwrite')
    prior=NEW/'independent_native_stepper_equivalence_v1';proposal=read(prior/'proposal.json');pins={}
    def add(path):
        path=Path(path).resolve();pins[path.as_posix()]={'path':path.as_posix(),'sha256':sha(path)}
    # Reuse the already verified native/NumPy package inventory, excluding unrelated
    # trace cases, inference packages and historical producer source trees.
    for entry in proposal['input_files']:
        path=Path(entry['path'])
        if '/codex_sonic_runtime/mjbatch323_20260910/venv/' in path.as_posix():
            if sha(path)!=entry['sha256']:raise ValueError('qualified runtime inventory changed')
            add(path)
    bundle=Path(proposal['native_bundle'])
    for name in ('manifest.json','native_prepared.xml','prepared_model_arrays.npz','contract.json','walk003/native_original.npz'):
        add(bundle/name)
    for name,digest in read(bundle/'manifest.json')['meshes'].items():
        path=bundle/'meshes'/name
        if sha(path)!=digest:raise ValueError('native geometry changed')
        add(path)
    roles={
        'expert_main':NEW/'bfm_entry250_actual_oracle_v1/nominal/trace.npz',
        'expert_hold':NEW/'bfm_entry250_actual_oracle_v1/post_lifecycle_hold_5s/trace.npz',
        'canonical_fixture':NEW/'walk003_canonical_initial_fixture_v1/initial_integration_state.npz',
        'reference':Path('E:/codex-artifacts/sonic23_teleop_six_hour_20260910/mjbatch_intent_floor_inputs_v1/walk003/reference.npz'),
        'mjb_witness_report':prior/'witness/report.json','expected_model_mjb':prior/'witness/expected_model.mjb',
        'native_saved_audit':NEW/'independent_native_stepper_saved_audit_v1/results_v1/report.json',
        'native_replay_owner':prior/'replay_completion_verification.json',
        'expert_qualification':NEW/'bfm250_expert_root_qualification_v1/qualification.json',
        'scaffold_review':NEW/'independent_plant_process_clock_root_review_v1/review.json',
        'mailbox_WSL_owner':NEW/'independent_plant_mailbox_transport_wsl_v1/owner_completion.json',
        'expert_main_physics':NEW/'bfm250_expert_full_independent_physics_v1/report.json',
        'expert_hold_physics':NEW/'bfm250_expert_hold_independent_physics_v1/report.json',
        'bootstrap':Path('Z:/codex/GR00T-WholeBodyControl-sonic-transfer-23dof/artifacts/teleop_six_hour_20260910/RUN_PASSING_WALK_WSL.sh')}
    for path in roles.values():add(path)
    expected={'expert_main':'721a44a88ba3c8d1c9a42c23ab3ec1d0b39b529c78abaa54df19f07ebf5a886a',
              'expert_hold':'e171f7e3cc6a1b08484f4f6d328bc43ebc4ff6a39555416157c4d5191d713780',
              'canonical_fixture':'4f99f1bd0d9559ec23bd0a73afe5e49a711a90c3dcecafd1a121bae89b0dbde3',
              'expected_model_mjb':'717cc6f01a61be2fd1e79bff25710b30de4d01d5f14511f0f76ae45168194ee4',
              'native_saved_audit':'66e6e1855b3f816b9834f11880847154d878913de708dcea177d1509b48717de',
              'scaffold_review':'6b65db0b41240de22d857784794e080015065c8821a28da89cdf0b2f22fcca51'}
    for role,digest in expected.items():
        if sha(roles[role])!=digest:raise ValueError('qualified original subject differs: '+role)
    for name,steps in [('expert_main',15690),('expert_hold',2500)]:
        report=read(roles[name+'_physics'])
        if report['feasible'] is not True or report['recorded_trace_reproduced_through_last_sample'] is not True or report['physics_steps']!=steps:
            raise ValueError('expert physical qualification differs')
    def archive(path):
        with np.load(path,allow_pickle=False) as values:return {k:values[k] for k in values.files}
    full=archive(roles['expert_main']);hold=archive(roles['expert_hold']);fixture=archive(roles['canonical_fixture'])
    if int(fixture['state_spec'])!=8191:raise ValueError('full291 fixture spec')
    table=canonical_table(full,hold,sha(roles['expert_main']),sha(roles['expert_hold']),
        np.asarray(read(bundle/'contract.json')['joint_limits'],np.float64),fixture['state_vector'])
    copies={}
    for original in (BASE/'source_draft_v3').glob('*.py'):
        copy=SOURCE/original.name
        if sha(copy)!=sha(original):raise ValueError('reviewed scaffold copy changed: '+copy.name)
        copies[copy.name]={'original_path':original.as_posix(),'sha256':sha(copy)}
    for name in ('native_loader.py','counted_api.py'):
        original=prior/'source_draft_v1'/name
        if sha(SOURCE/name)!=sha(original):raise ValueError('qualified native helper copy differs')
        copies[name]={'original_path':original.as_posix(),'sha256':sha(original)}
    for path in SOURCE.glob('*.py'):add(path)
    for name in ('freeze_packet.py','prepare_clock_stage.py','stage_verdict.py','verify_completion.py','OUTPUT_SCHEMA.md','runner_stub_tests.xml'):
        add(BASE/name)
    request={'kind':'recorded_command_clock_request','preparation_only':True,'execution_selected':False,
        'execution_requires_separate_root_clearance':True,'run_id':'query250-recorded-clock-v1','input_epoch':1,
        'requested_controls':1819,'main_controls':1569,'hold_controls':250,'native_step_budget':18190,
        'serialization_budget':4,'model_inference_calls':0,'optimizer_updates':0,'other_oracle_native_steps':0,
        'epoch_lead_ns':200000000,'epoch_rebase_allowed':False,'debt_abort_steps':100,'elapsed_abort_ns':60000000000,
        'outer_process_timeout_seconds':120,'native_bundle':bundle.as_posix(),'source_directory':SOURCE.as_posix(),
        'output_directory':(BASE/'run').as_posix(),'expected_model_sha256':expected['expected_model_mjb'],
        'command_table_sha256':table.sha256,'roles':{k:v.as_posix() for k,v in roles.items()},
        'byte_preserved_copies':copies,'input_files':[pins[k] for k in sorted(pins)],
        'independent_saved_trace_audit_required':True,'online_policy_qualification':False,'hardware_authorized':False}
    require_roles(request);write(request_path,request)
    print(json.dumps({'request_sha256':sha(request_path),'pins':len(pins),'command_table_sha256':table.sha256,
                      'native_steps':0,'model_calls':0,'worker_starts':0,'execution_selected':False}))


if __name__=='__main__':main()
